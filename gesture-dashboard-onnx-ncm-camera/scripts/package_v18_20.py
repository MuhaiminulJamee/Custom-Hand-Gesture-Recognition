"""Portable notebook/code/model bundle, excluding source photos and user data."""
from pathlib import Path
import zipfile


def package(root):
    root = Path(root)
    paths = []
    for folder in ['backend', 'research', 'scripts', 'models', 'artifacts/v18_20', 'data/v18_20']:
        for path in (root / folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.suffix in ['.zip', '.pyc', '.joblib', '.pkl'] or 'online_adapter' in path.name:
                continue
            if path.name == 'last_live_test.csv':
                continue
            paths.append(path)
    for name in ['v18_20.ipynb', 'V18_20_README.md', 'HAND_DISTANCE.md', 'requirements.txt', 'requirements-model-build.txt', 'requirements-notebook.txt', 'pytest.ini']:
        paths.append(root / name)
    target = root / 'v18_20_bundle.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, 'v18_20_bundle/' + path.relative_to(root).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Bundle verification failed')
    print(f'{target.name}: {target.stat().st_size / 1024**2:.1f} MiB; {len(paths)} files')


if __name__ == '__main__':
    package(Path(__file__).resolve().parents[1])
