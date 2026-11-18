# Eye Gaze & Visual Attention Observation

Local desktop research prototype for coarse gaze observation, timestamped intervals,
source-video review and separate therapist annotations. Built around the supplied
Saachivya Medical problem statement. No cloud processing during analysis.

## Progress as of 18 November 2026

The desktop prototype supports coarse directional gaze observations, timestamped intervals, video review and export. The local sessions folder contains eight MP4 recordings and a SQLite database. These files document local trials. Direction accuracy still needs a labelled comparison.

| Milestone | Current evidence | Next check |
| --- | --- | --- |
| Landmark baseline | Webcam recordings and local session storage exist. The app supports a heuristic mode without a trained checkpoint. | Run a scripted left, right, centre and face-away sequence. Compare predicted intervals with timestamped labels. |
| Review and export | Timeline review, separate annotations, and JSON/CSV export are implemented. | Follow the manual acceptance checklist and record pass/fail results. |
| CPU training | Gaze360 preprocessing, training and evaluation scripts are present. Earlier builds reported passing core and synthetic training tests. | Obtain the authorised dataset, inspect the split report and train the custom GazeNet. |
| Evaluation | Real-data angular error and room-level direction accuracy are pending. | Report held-out angular error in degrees and separately report direction results on labelled room recordings. |

Keep recordings, individual observations, datasets and model weights local. Share aggregate findings and reproducible code changes in commits.

## 1. Windows setup

Use Python 3.12 for this project, even if your system Python is 3.14. Open PowerShell
in this folder. `uv` is an external environment manager; do not run `python -m uv`
inside a venv that does not contain it.

```powershell
# Only if uv is not already installed:
python -m pip install uv

.\setup.ps1
.\.venv\Scripts\python.exe scripts/doctor.py
.\run.ps1
```

Setup creates an isolated Python 3.12 environment, installs app/training dependencies,
and downloads Google's MediaPipe face-landmarker asset. It does not download Gaze360
or start training. Internet is needed for setup only. If script execution is blocked,
run the equivalent commands manually without changing machine security policy:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-training.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe main.py
```

The face-landmarker model is separate from your gaze model. It detects face/eye
landmarks and supplies quality gates for both inference modes.

## 2. Dataset preparation (after you download it)

Obtain the official Gaze360 dataset through its authorised distribution. This
project does not bundle it or bypass access restrictions. Expected extracted layout:

```text
data/raw/gaze360/metadata.mat
data/raw/gaze360/imgs/<recording>/head/<six-digit-person>/<six-digit-frame>.jpg
```

The input is the official `metadata.mat` plus head-crop JPEGs, not an unrelated
CSV-only or restructured mirror. Point `--root` to the directory containing
`metadata.mat` (it may be on another drive):

```powershell
.\.venv\Scripts\python.exe scripts/preprocess_gaze360.py --root "D:\Datasets\gaze360" --output data/processed
```

Preprocessing projects each official face box into its head crop, resizes it to
128×128, normalizes finite 3D gaze vectors, checks eye annotations, preserves
official train/val/test membership, and rejects duplicate or cross-split identities.
Rear-facing samples and samples without usable face/eye annotations are excluded.
Missing/corrupt files are listed in `preprocessing_report.json`; inspect that report
before training. All three splits must be nonempty. An existing nonempty output
directory is rejected to prevent stale-data mixing.

This is a **visible-eye frontal subset**, not the full official benchmark.
Face-box differences between dataset annotations and MediaPipe at inference create
a domain gap. Validate with representative camera recordings after training.

## 3. Train on CPU

```powershell
.\.venv\Scripts\python.exe scripts/train.py --data data/processed --output runs/gaze360 --epochs 30 --batch-size 32 --threads 4
```

The custom `GazeNet` is a small single-frame CNN trained from scratch with cosine
vector loss. It is NOT a reproduction of the original temporal Gaze360 network and
cannot load that network's `.pth.tar` weights. No pretrained weights are downloaded.
Start with a smaller batch (8 or 16) if memory is constrained. Epoch time depends on
your CPU and accepted dataset size; full training may take substantial time.

Outputs: `best.pt`, `last.pt`, `config.json`, and `history.jsonl`. Validation mean
angular error selects the best checkpoint. Early stopping defaults to five epochs
without improvement. The test set is not used for model selection.

Resume interrupted training in the same run directory:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --data data/processed --output runs/gaze360 --resume runs/gaze360/last.pt --epochs 50
```

`--epochs` is the total target, not additional epochs. Resume restores model,
optimizer, CPU random state and loader random state from the last completed epoch.
Changing dataset manifests is rejected. A new experiment needs a new output folder.

Evaluate the selected model once on the held-out test subset:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --data data/processed --checkpoint runs/gaze360/best.pt
```

The report contains mean/median/90th-percentile angular errors in degrees and sample
count. Synthetic test metrics are not model accuracy results.

## 4. Use the app

1. Confirm consent and authority to process the video. Supply a shared session ID
   if coordinating with other observation algorithms; otherwise one is generated.
2. Optionally select your trained `best.pt`. Leaving this blank uses an uncalibrated
   landmark/head-pose heuristic, clearly labelled in session metadata.
3. Analyse a saved video or start the webcam. Live sessions record locally for
   subsequent review. Stop safely before closing the app.
4. Inspect summary metrics and click a timeline table row to jump to its video
   interval. Use Play/pause to review surrounding evidence.
5. Select a row and save an accepted/corrected/dismissed/annotated therapist review.
   Original machine output remains unchanged.
6. Export JSON for observations, summary, metadata and separate reviews. CSV exports
   machine observation rows. Sessions can be reopened from the history selector.

Directions use the unmirrored viewer image: left means image-left, not the child's
anatomical left. Looking towards camera does not establish eye contact.

CLI alternative:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_video.py "D:\Videos\session.mp4" --consent-confirmed --session-id S-2026-0142 --checkpoint runs/gaze360/best.pt --output observations.json
```

## Behaviour and limitations

- Bad lighting, blur, closed eyes, extreme pose, missing faces, multiple faces and
  unreliable eye geometry produce unavailable/low-confidence observations. A
  trained model never bypasses these gates. Heuristics cannot detect every occlusion
  or reflection; validate false-valid cases manually.
- A 150ms dwell confirms directions; brief unconfirmed changes remain unavailable.
  Blink intervals remain unavailable and are not counted as looking away. Shifts
  are not inferred across unavailable gaps.
- Recorded files use presentation timestamps via PyAV. Live capture uses a
  monotonic clock. Capture gaps exceeding 100ms become unavailable, not prolonged
  gaze claims. Recording playback may visually hold a frame across those gaps;
  the observation timeline marks the gap separately.
- Confidence is heuristic **tracking quality**, not a calibrated probability that
  the gaze prediction is correct. It is labelled High/Medium/Low.
- Summaries divide direction time by analysable time only. Unavailable time remains
  explicit; no attention score or diagnostic judgement is produced.
- One participant must be visible. Multiple faces are conservatively excluded;
  there is no person-identity selector or cross-person re-identification.
- Optional person/toy target calibration is not implemented. The app reports coarse
  directions only; it does not infer whom or what the child is looking at.
- Video review currently decodes from the beginning to a clicked timestamp; long
  recordings can take time to seek. Audio review is not implemented.

## Storage and privacy

SQLite lives in `sessions/observations.sqlite3`; webcam videos and error logs are
also in `sessions/`. Saved input videos remain at their original paths. If moved,
restore them to that path for evidence review. Deleting a session removes its database
observations/reviews, not source videos. Delete recordings separately under your
retention policy. Data is local but **not encrypted** by the app; use authorised OS
accounts and suitable disk/access protection. No videos or datasets belong in Git.

## Tests and troubleshooting

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q gaze scripts tests main.py
```

Automated tests cover interval denominators, missing gaps, blink continuity, shift
timestamps, duration-weighted quality, review separation, exports, coordinate signs,
official-format synthetic preprocessing, a synthetic train/resume cycle and errors.
`tests/MANUAL_ACCEPTANCE.md` lists integration and real-video checks still required.

| Failure | Action |
| --- | --- |
| `No module named uv` inside venv | Use the standalone `uv` command and explicit `--python` path. |
| `pypi.org` DNS error | Restore internet/DNS access and rerun setup. |
| Missing face model | Run `scripts/download_model.py` with the project Python while online. |
| Gaze360 HTTP 403 | Contact the official dataset maintainers; permission to use the dataset does not fix server denial. |
| Empty preprocessing split | Check extraction root and rejection report; do not invent labels or random frame splits. |
| Camera cannot open | Check camera index, Windows camera privacy permission and competing apps. |
| Multiple faces / small face | Isolate the participant and improve camera framing. |
| Unsupported checkpoint | Use this project's `best.pt`; original Gaze360 checkpoints are a different architecture. |
| Missing video timestamps | Convert a working copy to a timestamped MP4 before analysis. |

## Sources

- Requirements: supplied `PS1_Eye_Gaze_Visual_Attention.pdf`.
- Dataset format/coordinates/splits: https://github.com/erkil1452/gaze360/blob/master/dataset/README.md
- Official download: https://gaze360.csail.mit.edu/download.php
- Dataset licence: https://github.com/erkil1452/gaze360/blob/master/LICENSE.md
- Face model asset: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

The official Gaze360 repository states non-commercial research use. Check its
licence before using it for an organisation's deployment. A clinician must validate
the observation protocol and outputs before any use in care decisions.
