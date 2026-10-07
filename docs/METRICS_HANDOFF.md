# Evaluation metrics — handoff brief

**Written for:** Yuval, and whatever coding agent she works with.
**Date:** 2026-10-06
**Scope:** build content and timbre metrics that work for **both** one-instrument→one-instrument and two-instrument→two-instrument transfer, for **any** instrument in CocoChorales, staying as close to the paper as the two-instrument case allows.

Everything in `evaluation/` stays. This is additive.

---

## 1. Why this is needed

The paper (§III-E) reports **four** metrics. We have two.

| Paper's metric | Status here |
|---|---|
| DPD — dynamic pitch distance | ✅ implemented, upstream code |
| JD — Jaccard distance | ✅ implemented, upstream code |
| **Instrument classifier accuracy** | ❌ **missing entirely** |
| **FAD on EnCodec embeddings** | ⚠️ we have FAD, but on CLAP — different scale, not comparable to the paper |

And the two we have are **structurally unable to score a two-instrument mixture**, for a reason that is one line of code. See §3a.

The missing classifier matters more than it looks. In the paper's Table I it is what exposed that the GFB baseline had failed: GFB's DPD looked excellent, and its timbre accuracy was **0%** — it had preserved the melody and changed nothing about the sound. DPD alone cannot see that failure, and it is a plausible failure mode for us.

---

## 2. Decisions already made — please don't re-litigate these

| Decision | Reason |
|---|---|
| **DPD and JD stay exactly as they are** | They are the paper's numbers. Changing them breaks comparability with Table I. Report them, caveat them for mixtures, don't base decisions on them. |
| **Primary content metric: `mir_eval.transcription`** | The only option that is both polyphonic **and** time-tolerant. It matches note events and counts a match when the pitch is close enough and the onset falls within 50 ms. Field standard, handles 1 and K voices identically. |
| **Secondary: `mir_eval.multipitch`** | Frame-by-frame, no time tolerance at all. Nearly free, since both sides are already frames. Its value is the **gap** against `transcription` — see §4a. |
| **Compare against the dataset's ground-truth MIDI**, not against a transcription of the input | Every CocoChorales track ships `stems_midi/`. Transcribing both sides stacks two error sources; using the real score removes one for free. |
| **Report P and R separately, not only F1** | Low recall = notes lost. Low precision = notes invented. At σ_max=100 the model generates from near-pure noise, which is exactly when a diffusion model invents plausible notes that were never played. F1 blurs the two; DPD cannot see the distinction at all. |
| **Bounded time tolerance, not DTW** | 50 ms forgives the small drift a diffusion model may introduce. DTW's flexibility is *unbounded* — it will happily align a note at 3 s to one at 9 s if that lowers the total cost, and a note moved six seconds is not a melody that survived. See the rejected alternative at the end of §4a. |
| **FAD: compute on EnCodec ourselves** | No `fadtk` install. Absolute values won't match the paper either way; we only need to rank our own runs. Keep the CLAP FAD too and report both. |
| **Add the paper's classifier, generalised to multi-label** | See §4c. |
| **200 clips** | Enough for FAD to be meaningful; DPD needs far fewer. |

---

## 3. Findings from reading the code — these will save hours

Each of these was verified against the source. Line numbers are current as of commit on `fix/midi-alignment-and-cache-cleanup`.

### a. The polyphony is computed and then thrown away

`basic-pitch` returns a full polyphonic `[T, 88]` note matrix. The metric code reduces it in two steps:

- `aggregate_to_semitones` ([`pitch_tracking_utils.py:21-29`](../audio_diffusion_pytorch/pitch_tracking_utils.py#L21)) folds 88 keys to 12 pitch classes with a **max** over octaves. Polyphony survives this.
- `extract_dominating_melody` ([`:30-37`](../audio_diffusion_pytorch/pitch_tracking_utils.py#L30)) then does `np.argmax(semitone_array, axis=-1)` — **one note per frame**. Everything else is gone.

So when a flute and a bassoon play different notes simultaneously, only the louder is scored. A model that preserved one voice and produced garbage for the other scores well.

**The information you need already exists** one line earlier. You do not need a different transcriber.

### b. `specs_back` does not give you the raw 88

It is tempting to assume the `specs_back=True` flag returns the unfolded matrix. It does not — the fold at [`:113-114`](../audio_diffusion_pytorch/pitch_tracking_utils.py#L113) happens before every return path, and `specs_back` returns `note_pt1_trimmed`, which despite the name is the `[B, T, 12]` folded array. The raw locals are never returned. `specs_back` is also referenced nowhere else in the repo.

Get the raw matrix by calling the model directly:

```python
from audio_diffusion_pytorch import PitchTracker
tracker = PitchTracker()              # handles the weight loading and sys.path juggling
out = tracker.pt_model(wav)           # wav: [B, 1, N]
note = out['note'][0].cpu().numpy()   # [T, 88], probabilities in [0,1]
```

Row `i` is MIDI note `21 + i` (A0 = 21). The repo's piano rolls are `[128, T]` indexed by raw MIDI number, so rows `21:109` line up.

### c. ⚠️ Sample-rate mismatch — load-bearing for you

`cals_pitch_metric` passes the waveform to the model with **no resampling** ([`:106`](../audio_diffusion_pytorch/pitch_tracking_utils.py#L106)). Everything in this repo is 24 kHz; basic-pitch expects **22050 Hz**.

Consequences at 24 kHz input:
- every pitch reads **≈1.47 semitones sharp** (`12·log2(24000/22050)`)
- one output frame is `256/24000 s` → **93.75 fps**, not the nominal 86.13

**For DPD and JD this cancels** — both sides go through the identical wrong path, so the existing CSV numbers are internally consistent and should not be "fixed".

**It does not cancel for anything compared against an absolute reference.** Scoring a transcription against `stems_midi/*.mid` without resampling to 22050 first will be off by a semitone and a half in pitch and ~9% in time. Resample in the new code; leave `pitch_tracking_utils.py` alone.

### d. The input audio is not archived anywhere usable

[`Run_timbre_transfer.py:396-398`](../Run_timbre_transfer.py#L396) writes the cropped input to a **single fixed path** with no `sample_id` and no `experiment` in the filename. Every run overwrites it. After a 20-track sweep only the last input survives, and the CSV has no column pointing at it.

To get the paired input you must **re-derive it from `sample_id`**, replicating [`Run_timbre_transfer.py:357-388`](../Run_timbre_transfer.py#L357) exactly:

1. load each stem, resample to 24 kHz
2. **sum** the stems
3. peak-normalise to 0.95 **only if** there is more than one stem **and** the peak exceeds 0.95
4. crop `[..., :409600]` from the **start** (deterministic — unlike training, which crops randomly), zero-pad if short

Get any of these wrong and your reference is not frame-aligned with what the model actually saw.

Alternatively, have your own runner archive the input alongside the output with a matching `{experiment}_{sample_id}_` prefix. Cleaner, and you need a runner anyway (§5).

### e. The metrics CSV header is frozen by its first row

[`Run_timbre_transfer.py:522`](../Run_timbre_transfer.py#L522) does `write_header = not metrics_path.exists()` then `csv.DictWriter(fieldnames=row.keys())` in append mode. Adding a column to that dict later silently misaligns every row written afterwards against the old header.

**Write a separate CSV and join on `(experiment, sample_id)`.** Do not extend `inference_metrics.csv`.

Its current columns:

```
timestamp, experiment, sample_id, source, target, output_audio, DPD, JD, num_steps
```

- `sample_id` is the CocoChorales track folder name — this is the pivot key `paired_stats.py` already uses
- `source` / `target` are ensemble names (`flute_bassoon`), not per-stem lists
- `output_audio` is a **relative** path, so consumers must run from the repo root

### f. `sigma_handoff` is hidden inside the experiment string

There is no column for it. [`Run_timbre_transfer.py:137-149`](../Run_timbre_transfer.py#L137) appends it to the experiment name instead:

```python
if SIGMA_HANDOFF != TRAINED_SIGMA_MAX:
    EXPERIMENT_NAME = "%s_h%g" % (EXPERIMENT_NAME, SIGMA_HANDOFF)
```

So a sigma sweep produces `original_midi_s100_h20`, `original_midi_s100_h5`, … and `prepare_fad_sets.py` currently prints `Skipping unknown experiment` for all of them. Your runner should log it as a proper column.

### g. EnCodec standalone — no checkpoint needed

```python
from audio_diffusion_pytorch import NormalizedEncodec
enc = NormalizedEncodec(device="cuda")
z = enc.encode_latent(wav_24k, None, None)   # [B, 128, 1280] for a 409600-sample clip
```

`mean`/`std` are `Optional` and the normalisation is gated on both being non-`None`, so `None, None` gives the raw latent.

**Do not normalise for FAD.** The `{violin_cello,flute_bassoon}_{mean,std}.pt` tensors are per-ensemble and differ measurably (std max 2.175 vs 2.376). FAD compares distributions *across* ensembles by construction; normalising each by its own statistics would destroy exactly the difference being measured.

Do **not** go through `Model.encode_latent` — that requires instantiating the full LightningModule, which needs a diffusion net and builds a 604M-parameter EMA wrapper, all to use a 14.9M encoder.

### h. The Fréchet computation is already written and reusable

[`evaluation/compute_clap_fad.py:62-85`](../evaluation/compute_clap_fad.py#L62) is a self-contained `frechet_distance(x, y)` on `scipy.linalg.sqrtm` taking `[n_clips, D]` arrays. **Import it, don't copy it.** Swapping CLAP for EnCodec is a change of embedder, not of metric. You will need to pool `[128, 1280]` → one vector per clip; mean over time is the usual choice.

### i. basic-pitch's own error rate, measured on our data

A comparison run on one monophonic cello stem, ground truth vs basic-pitch:

| | ground truth | basic-pitch |
|---|---|---|
| notes | 30 | **62** |
| range | D3–E4 | E2–B4 |
| simultaneity | strictly 1 | up to 3 |

IoU 0.830 — 69 frames where a real note was missed, **281 frames where a note was reported that was never played**. Mostly octave doublings.

**So the transcriber roughly doubles the note count on monophonic material.** This is why the ceiling measurement in §4b is not optional. Without it, an absolute F1 number is uninterpretable: part of the gap is basic-pitch, not the model.

### j. There is no instrument inventory in this repo

The only list is `instrument_to_idx` ([`main/module_base_latent_cond.py:74-88`](../main/module_base_latent_cond.py#L74)) — 13 instruments. Three caveats:

1. It lives inside `elif self.cond == "label"`, and **every** experiment config sets `cond: False`. It has never been constructed in any run, so it has never been validated against real filenames.
2. ~~`'double bass'` contains a space and so matches nothing.~~ **Corrected 2026-10-07:** the stem filenames also contain a space (`4_double bass.wav`), so that key is fine.
3. It is a name→index map with no stem-number prefix (`1_`, `4_`), and the prefix varies by family.

**Enumerated from disk 2026-10-07, so the list below is now fact rather than assumption.** 31,255 tracks, four stems each (125,019 stems), 13 instruments:

| ensemble | tracks | | instrument | stems | | instrument | stems |
|---|---|---|---|---|---|---|---|
| woodwind | 8,000 | | violin | 18,658 | | bassoon | 9,982 |
| string | 8,000 | | clarinet | 11,909 | | trumpet | 9,907 |
| random | 8,000 | | cello | 11,395 | | horn | 9,557 |
| brass | 7,255 | | oboe | 10,551 | | tuba | 9,228 |
| | | | flute | 10,541 | | trombone | 8,572 |
| | | | viola | 10,374 | | saxophone | 2,337 |
| | | | | | | double bass | 2,008 |

Tracks holding **both** instruments of a pair: violin+cello **8,941**, flute+bassoon **8,586**. These match the counts in CLAUDE.md exactly.

Note the counts exceed the ensemble sizes: a string quartet has two violins, and the `random` ensemble draws from all 13 — which is where saxophone and double bass come from, and why they are an order of magnitude rarer.

The command, if it needs rerunning:

```bash
D=/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset
ls -d $D/*_track* | sed 's#.*/##; s/_track.*//' | sort | uniq -c      # families
ls $D/*_track*/stems_audio/*.wav | xargs -n1 basename | sort -u       # exact stem filenames
```

The MIDI cache lookup (`wav_dataset.py`) and `create_gt_midi_cache.py --instrument` are already fully generic, so no code changes are needed to support a new instrument there.

### k. Dependencies

**Installed and working in `venv_yuval`:** `pandas`, `scipy`, `transformers`, `librosa`, `torchaudio`, `wandb`, `fastdtw`, `nnAudio`, `pretty_midi`.

**Not present anywhere — neither declared nor imported:** `mir_eval`, `fadtk`, `frechet_audio_distance`, `scikit-learn`.

`pip install mir_eval` into the venv is the only new dependency the plan needs.

### l. ⚠️ The evaluation data overlaps the training data

`main_dataset/` contains 31,255 extracted track folders. It also contains `train/`, `valid/`, `test/` subfolders holding un-extracted `.tar.bz2` archives (77G / 26G / 26G).

**The extracted folders are a mix of all three splits.** Verified: `string_track216223` appears inside `test/1.tar.bz2` **and** exists as an extracted folder. Extracted track IDs run up to 236000, above the test archive's range.

So the models were trained on test-split data, and `evaluation/track_list_20.txt` (`woodwind_track096001`–`096020`) is drawn from the same pool — with an 80/20 split, most of those 20 were probably training examples.

**Checked, and the answer is the bad one.** The test archives list 8,004 tracks. Every single one of them is already extracted:

```bash
comm -23 test_tracks.txt extracted_tracks.txt | wc -l
0
```

**There is no unseen data.** The entire official test split was absorbed into the training pool. A held-out evaluation is impossible without retraining all sixteen models on a restricted file list, which is not worth it.

**What this does and does not invalidate:**

- ✅ **Comparisons between conditions remain fully valid.** Both conditions see identical clips, so the difference between them is a fair measurement. Every conclusion this project has drawn rests on differences, not absolute values.
- ❌ **Absolute numbers are optimistic**, and we cannot write "evaluated on held-out data". This needs to be stated plainly in the report — a reviewer will ask.

Practical consequence for the metrics work: none. Build and run them as planned. Just don't describe the evaluation set as unseen.

---

## 4. What to build

### a. Polyphonic content metric — the primary number

**`mir_eval.transcription.precision_recall_f1_overlap`**, output transcription vs ground-truth MIDI.

It is the only standard metric that is **both** polyphonic and time-tolerant: it converts both sides to note events, then matches them, counting a note correct when the pitch is within tolerance **and** the onset falls within 50 ms. Several notes at the same instant are fine — the matching is note-level, not frame-level.

**`mir_eval` needs no MIDI files.** It takes plain numpy arrays — `(onset, offset)` intervals and pitches in Hz. The file format never enters into it.

Where each side comes from:

| side | source | cost |
|---|---|---|
| reference | `stems_midi/*.mid` of the **input** track | free — `pretty_midi` already returns note objects with onset, offset and pitch |
| estimate | basic-pitch on the **generated** audio | needs frames → note events: threshold, then merge contiguous runs. The original basic-pitch library does this; check whether the vendored PyTorch port exposes it |

The output can only ever be transcribed — it is generated audio, there is no score for it. That is equally true of DPD today. **What changes is the reference side:** DPD transcribes the input too, so transcription error is counted twice and we compare two noisy readings. The dataset's MIDI is an exact reading, and removing one of the two error sources costs nothing.

Settings and details:

- resample audio to **22050** before transcription (§3c)
- use the raw `[T, 88]` — **no octave folding, no argmax** (§3a, §3b)
- for K instruments, **merge the K ground-truth rolls** — the question is "should this note be sounding, from any instrument", not which instrument played it
- **set `offset_ratio=None`** so note endings are ignored. A diffusion model has no obligation to preserve note durations, and scoring them hides what we actually care about
- report **Precision, Recall and F1 separately**. Low recall = notes lost. Low precision = notes invented. At σ_max=100 the model generates from near-pure noise, which is exactly when a diffusion model invents plausible notes that were never played
- calibrate basic-pitch's confidence threshold on real recordings where the truth is known, rather than guessing

Identical code path for 1 and K instruments; the only difference is how many ground-truth rolls get merged.

**Attribution is explicitly out of scope.** Deciding which instrument played a detected note is source separation — much harder, and no metric in the paper attempts it.

#### Why `multipitch` as well

It needs no note-event conversion — both sides are already frames — so it is nearly free once the rest is built. But it has **no time tolerance whatsoever**; frames are matched by index.

That strictness is the point. Run both and compare:

| | reading |
|---|---|
| the two roughly agree | timing is intact; the strict number is trustworthy on its own |
| `multipitch` much lower than `transcription` | **the model is shifting events in time** — a finding in its own right about what high σ does |

Two numbers, two questions. That is cheaper and more informative than trying to build one metric that answers both.

#### An alternative that was considered and rejected

Warping the two `[88, T]` matrices against each other with multi-dimensional DTW *first*, then scoring the aligned pair with `multipitch`.

It is technically sound — DTW works on sequences of vectors as readily as scalars — but it has a disqualifying flaw. **DTW finds the alignment that minimises the distance.** Scoring what remains after that means measuring the residue of something an optimiser has already worked to shrink; a model that badly scrambles its timing gets handed a warp path that makes it look fine.

The timing information lives in the **cost of the warp path**, and that approach computes it and then discards it. DPD, whatever its other faults, does not make this mistake — the warp cost *is* what it reports.

If flexibility beyond 50 ms is ever wanted, the right form is to report the warp cost as its own separate number alongside the content score, so that "wrong notes" and "right notes in the wrong place" stay distinguishable.

### b. The transcription ceiling — not optional

Transcribe the **real input recordings** and score them against their own ground truth, per instrument.

That number is basic-pitch's error rate on exactly this material. Every model score is read against it. If the ceiling is 0.78 and a model scores 0.70, it is near the limit of what the tool can measure — not 30% wrong.

Per instrument matters: basic-pitch is not equally accurate on cello and bassoon, so a bassoon→cello bridge could look better than a cello→bassoon one purely because of the transcriber.

### c. Timbre classifier — the paper's, generalised

Paper's version: two fully-connected layers, 128→64→5, on EnCodec embeddings averaged over time, single-label, accuracy reported.

Generalisation needed for mixtures: **multi-label** — sigmoid outputs, one per instrument.

| case | what is checked |
|---|---|
| 1→1 | the one correct instrument fires |
| 2→2 | **both** target instruments fire |

Same model, same code, both cases.

Training data is free: take stems from the dataset; for mixtures, sum two stems from the same track. Labels are known by construction. The model is tiny and trains in minutes.

**Validate it on held-out real stems before pointing it at generated audio.** A classifier that cannot identify real instruments says nothing about synthetic ones.

### d. FAD on EnCodec

Reuse `frechet_distance` (§3h), swap the embedder to EnCodec (§3g), mean-pool over time. Keep the CLAP FAD alongside and report both, clearly labelled — they are on different scales and must never be compared to each other.

**Sanity check:** real vs real from the same ensemble should come out near 0.

---

## 5. A batched runner is worth building

`Run_timbre_transfer.py` loads **two 4.9 GB checkpoints plus EnCodec from scratch for every single clip**, and processes one clip at a time.

At 200 clips × 8 conditions × ~3 min that is **80 GPU-hours**.

A runner that loads once and processes `[16, 128, 1280]` batches cuts this by roughly an order of magnitude — training itself ran at batch 16 *with* gradients, so inference without them has ample room.

**The risk** is two bridge implementations drifting apart, after which you are comparing different things. **The mitigation is cheap and decisive:** the pipeline is fully deterministic (`s_churn` is forced to 0 in both samplers), so run one clip both ways and assert the outputs match exactly. If they do, it is the same bridge.

Log the columns the existing CSV lacks: input stem paths, checkpoint paths, and `sigma_handoff` as a real column.

---

## 6. Model inventory — what can actually be evaluated

All 85 epochs, batch 16.

| | σ=5, MIDI | σ=5, no MIDI | σ=100, MIDI | σ=100, no MIDI |
|---|---|---|---|---|
| flute+bassoon | 0.586 | 0.592 | 0.593 | 0.667 |
| violin+cello | 0.632 | 0.637 | 0.612 | 0.699 |
| cello | 0.588 | *training* | 0.599 | *training* |
| bassoon | 0.531 | *training* | 0.546 | *training* |

The four single-instrument no-MIDI runs were launched 2026-10-06 and finish in ~10 hours. Their `runs/.hydra/overrides.yaml` were checked after launch and all four carry the right `exp=` and `sigma_max=` — worth repeating for any future batch, since a mis-pasted launch produces a correctly-named directory holding the wrong instrument, and nothing downstream would catch it. After that the matrix supports **8 bridges**: flute+bassoon→violin+cello and cello→bassoon, each with and without MIDI, at each σ.

⚠️ **`valid_loss` is not comparable across σ.** The loss is σ-weighted and drawn from a different noise distribution — the σ=100 models see a median σ roughly 10× higher. Comparing 0.667 to 0.592 is meaningless. Within a σ, MIDI vs no-MIDI is a fair comparison, and it is large: **~12% at σ=100 versus ~1% at σ=5**. Confirming that in DPD/FAD is the main thing these metrics are for.

Other checkpoints in `our_checkpoints/` are not comparable: `*_40ep` used the old single-injection architecture, `*_nofifth_v1` stopped at epoch 3, `*_tiny_v2` at epoch 1, `*_add_fifth_*` are Phase 1.

---

## 7. Do not

- **Modify `pitch_tracking_utils.py`.** Its quirks cancel for DPD/JD; "fixing" the sample rate there would change every existing number.
- **Modify `evaluation/paired_compare.py`, `summarize_metrics.py`, `prepare_fad_sets.py`, `compute_clap_fad.py`.** Add files alongside.
- **Extend `inference_metrics.csv`.** Frozen header (§3e).
- **Normalise EnCodec latents for FAD** (§3g).
- **Try to attribute notes to instruments.** Out of scope (§4a).
- **Compare CLAP FAD to EnCodec FAD**, or either to the paper's numbers.

---

## 8. Verification

| step | check |
|---|---|
| Transcriber shape | `tracker.pt_model(torch.zeros(1,1,22050))` returns `note: (1, T, 88)` |
| Batched runner | one clip, batched vs `Run_timbre_transfer.py`, outputs identical (pipeline is deterministic) |
| Content metric | feed the **input** as if it were the output — F1 should be high but not 1.0; that value **is** the ceiling |
| Classifier | accuracy on held-out **real** stems, before touching generated audio |
| FAD | real vs real, same ensemble → near 0 |
| Significance | run `evaluation/paired_stats.py` on the new metrics; 200 pairs resolve ~3× finer than 20 |

---

## 9. Existing files worth reading first

| file | what it is |
|---|---|
| `evaluation/run_compare_20.sh` | the driver loop — 20 tracks × 2 conditions |
| `evaluation/paired_compare.py` | per-track paired deltas — the design is sound, keep it |
| `evaluation/paired_stats.py` | sign test, Wilcoxon, 95% CI, and what effect size the run *could* have resolved. Experiment names are arguments, so it works for the new conditions too |
| `evaluation/compute_clap_fad.py` | `frechet_distance` lives here |
| `Run_timbre_transfer.py` | the bridge. Env-driven: `TRACK_NAME`, `EXPERIMENT_NAME`, `SIGMA_HANDOFF` |
| `audio_diffusion_pytorch/pitch_tracking_utils.py` | DPD/JD, upstream from the paper. Read it, don't edit it |
| `CLAUDE.md` | full project history, bug log, environment notes |
