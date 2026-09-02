from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.artifact_integrity import ArtifactRegistry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a SHA-256 manifest for trusted gesture runtime artifacts."
    )
    parser.add_argument("--models-directory", type=Path, default=PROJECT_ROOT / "models")
    arguments = parser.parse_args()
    registry = ArtifactRegistry(arguments.models_directory.resolve())
    manifest = registry.build()
    print(f"Manifest: {registry.manifest_path}")
    print(f"Release fingerprint: {manifest['release_fingerprint']}")
    print(f"Registered artifacts: {len(manifest['artifacts'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
