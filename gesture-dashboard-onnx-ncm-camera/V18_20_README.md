# v18_20

Open `v18_20.ipynb` in this project with the `.venv` Python kernel. It contains
executed metrics, count and normalized confusion matrices, per-class scores,
ROC/PR curves, learning curves, subgroup coverage, latency, and an optional live
camera test. Install notebook dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

For Colab, upload `v18_20_bundle.zip`, extract it under `/content` (it contains a
`v18_20_bundle/` folder), open the included notebook, and enable
`INSTALL_REQUIREMENTS`. To test your browser webcam there, set
`RUN_LIVE_CAMERA = True` and `LIVE_SOURCE = "colab"`. For local NCM testing use
`LIVE_SOURCE = "ncm"` with this project's backend running on port 8200.

## What changed

- New MLP trained on 36,864 balanced rows: 4,096 per command plus 4,096 unknowns.
- Real public HaGRID landmarks and 2,000 extracted Dorsal hands; training-only
  augmentation, with original source and parent example IDs retained.
- Imported public splits have 5,485 training, 686 validation, and 794 test
  participants, with zero overlap. Legacy replay has unknown participant IDs.
- A clear four-finger downward Dorsal pose can now correct any model label.
  Independent geometry evidence requires a longer 0.35-second hold if the model
  says unknown. Reviewed negative-feedback support still prevents that fallback.
- Confident pose transitions and capture outages require fresh confirmation;
  landmark smoothing accounts for elapsed time. Backend inference remains capped
  at 10 FPS even when capture delivers faster frames. ONNX uses one CPU thread.

The user's recorded 80-frame Dorsal failure originally rejected every frame.
After correction, replay confirms Dorsal after 0.4 seconds and on the remaining
76 frames. This is a development-case replay, not an independent population test.

## Evidence and limits

Public test: 98.79% correct known-command acceptance, 99.01% runtime macro-F1,
99.90% precision among accepted commands, and 2/500 unknown false accepts.
Dorsal: 331/331 correct public-test observations. The old cache has small
regressions, and its exact-overlap audit is shown in the notebook.

A 150-frame live timing run achieved approximately 9.9 FPS, pipeline p95 18.6 ms,
and one startup frame over 100 ms. That run has no truth labels and therefore is
not an accuracy measurement. Frame-rate simulations cover 5–60 FPS inputs.

Directional public examples are rotations of `one` hands. Augmentations do not
add new participants or demonstrate robustness to real lighting, occlusion,
children, disabilities, or every demographic. The 2-D model cannot independently
verify palm-versus-back appearance. Closely matching neutral downward-hand poses
can still trigger the Dorsal geometry rule; validate live false actions too.

## Public sources

- [HaGRID](https://github.com/hukenovs/hagrid): official landmarks and subject IDs;
  the authors' custom CC BY-SA 4.0 terms are linked in `data/v18_20/sources.json`.
- [Dorsal Hands](https://www.kaggle.com/datasets/mdmuhaiminulislam/dorsal-hands-dataset),
  version 1: CC BY-NC-SA 4.0.
- [11K Hands](https://github.com/mahmoudnafifi/11K-Hands), Mahmoud Afifi (2019):
  original participant metadata, obtained from the public Kaggle mirror recorded
  in `sources.json`. Its dataset use terms still apply to derived data.

The portable bundle excludes downloaded source image archives and local feedback.
`artifacts/v18_20/baseline/` preserves the prior model. Reproduction uses
`scripts/collect_v18_20_data.py`, `research/v18_20.py`, and
`scripts/qualify_v18_20.py`; all settings, fingerprints, audit results, and metrics
are saved under `artifacts/v18_20/`.
