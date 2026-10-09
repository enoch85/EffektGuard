#!/usr/bin/env bash
# Build the same HACS distribution in validation and manual tag releases.
set -euo pipefail
python -m compileall -q custom_components/effektguard
python - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path('custom_components/effektguard')
with ZipFile('effektguard.zip', 'w', ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            archive.write(path, path.relative_to(root.parent))
    for name in ('README.md', 'LICENSE', 'hacs.json'):
        archive.write(name)
with ZipFile('effektguard.zip') as archive:
    assert archive.testzip() is None
    assert 'effektguard/manifest.json' in archive.namelist()
PY
