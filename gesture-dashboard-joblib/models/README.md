# Trusted v18_17 Joblib artifacts

This folder must contain exports from your own trusted v18_17 notebook run. Never
load an untrusted Joblib/pickle file.

Recommended live-PC files:

- `hand_landmarker.task`
- `gesture_joblib_runtime_config.json`
- `mediapipe_hagrid_rps_pinch15_hagrid_fist_follow_object_models.joblib`
- `gesture_online_state.joblib` when OnlineMLP contains the latest feedback

Required only for PC online updates:

- `gesture_online_replay_cache.npz`
- `gesture_online_validation_cache.npz`
- `gesture_online_confirmed_features.npz` (optional prior history)
- `gesture_online_untouched_test_cache.npz` (evaluation archive)

Optional Analytics inputs are the notebook's `*model_comparison.csv`,
`*test_metrics.csv`, `*classification_report.csv`, and `*mlp_history.csv` files.

Use `scripts\import_models.ps1` as documented in the project README. The loader
rejects an old 14-class configuration; v18_17 must contain 15 classes with `fist`
at index 11 and `no_gesture` last.

This folder intentionally contains only the Python/Joblib gesture classifiers and
their required runtime support files.
