from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.artifact_integrity import ArtifactRegistry


manifest = ArtifactRegistry().build()
print(json.dumps({
    "manifest": str(ArtifactRegistry().manifest_path),
    "release_fingerprint": manifest["release_fingerprint"],
    "artifacts": sorted(manifest["artifacts"]),
}, indent=2))
