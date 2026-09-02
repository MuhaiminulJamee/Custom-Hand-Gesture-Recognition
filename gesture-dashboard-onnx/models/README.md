# Runtime artifacts

This directory is an atomic ONNX deployment unit. Required files are the ONNX
classifier, parity metadata, runtime configuration, MediaPipe task, and generated
artifact manifest.

Do not add a Joblib or TFLite fallback to this folder. The application is designed
to fail closed when the signed ONNX contract is incomplete or changed.

To approve an intentionally replaced, parity-qualified release, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_artifact_manifest.py
.\.venv\Scripts\python.exe .\scripts\onnx_preflight.py
```

Only rebuild the manifest after independently validating the new model and
metadata; the manifest is an integrity mechanism, not a quality test by itself.
