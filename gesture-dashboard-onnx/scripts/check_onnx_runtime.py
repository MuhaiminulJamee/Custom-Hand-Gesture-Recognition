from __future__ import annotations

import os
import platform
import struct
import sys
from pathlib import Path


if struct.calcsize("P") * 8 != 64:
    raise SystemExit("A 64-bit Python installation is required for this ONNX package.")

_dll_handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for value in (
        os.getenv("GESTURE_MSVC_RUNTIME_DIR", ""),
        str(Path(sys.base_prefix)),
        str(Path(sys.base_prefix) / "Library" / "bin"),
        str(Path(sys.executable).resolve().parent),
    ):
        directory = Path(value).expanduser() if value else None
        if directory is None or not directory.is_dir():
            continue
        try:
            _dll_handles.append(os.add_dll_directory(str(directory.resolve())))
        except OSError:
            continue

try:
    import onnxruntime as ort
except (ImportError, OSError) as error:
    raise SystemExit(
        "ONNX Runtime could not load. Install the Microsoft Visual C++ "
        "2015-2022 x64 Redistributable, reopen the terminal, and rerun setup. "
        f"Original error: {error}"
    ) from error

providers = ort.get_available_providers()
if "CPUExecutionProvider" not in providers:
    raise SystemExit(f"CPUExecutionProvider is unavailable. Providers: {providers}")

print(f"Python: {sys.version.split()[0]} ({platform.machine()})")
print(f"ONNX Runtime: {ort.__version__}")
print(f"Execution provider: CPUExecutionProvider")
