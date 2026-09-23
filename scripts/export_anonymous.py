#!/usr/bin/env python3
"""Export only distributable source/docs, excluding identity-bearing local state."""
import argparse
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOP_FILES={'README.md','THIRD_PARTY_NOTICES.md','pyproject.toml','run.py','.gitignore'}
TOP_DIRS={'scaffold_gnn','configs','scripts','RelatedMethods','docs','tests'}
EXCLUDED={'.git','.local','__pycache__','.pytest_cache','results','data','cache','logs','build','dist','.venv'}
BINARY_SUFFIXES={'.pyc','.pyo','.pt','.pth','.npy','.npz','.so','.o','.a','.log','.out','.pid'}


def source_files():
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or p.is_symlink():
            continue
        if rel.parts[0] not in TOP_DIRS and str(rel) not in TOP_FILES:
            continue
        excluded_parts=EXCLUDED-{'data'} if rel.parts[:2]==('scaffold_gnn','data') else EXCLUDED
        if any(part in excluded_parts or part.endswith('.egg-info') for part in rel.parts):
            continue
        if p.suffix in BINARY_SUFFIXES:
            continue
        yield p,rel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'dist/scaffold-anonymous.zip')
    args=p.parse_args()
    files=list(source_files())
    # Refuse personal home/scratch paths in text; third-party attribution stays.
    path_pattern=re.compile(r'/(?:qfs/)?people/[a-zA-Z0-9_.-]+/|/rcfs/scratch/[a-zA-Z0-9_.-]+/')
    for file,rel in files:
        if file.suffix in {'.png','.pdf','.gif'}:
            continue
        content=file.read_text(errors='replace')
        if path_pattern.search(content):
            raise SystemExit(f'Personal absolute path found in {rel}; remove it before exporting.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for file,rel in files:
            info=zipfile.ZipInfo(str(Path('Scaffold-GNN')/rel),date_time=(2026,1,1,0,0,0))
            info.external_attr=(0o100755 if file.suffix=='.sh' else 0o100644)<<16
            archive.writestr(info,file.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    print(f'{args.output.resolve()} ({len(files)} files; no Git history, data, results, or local overrides)')


if __name__=='__main__':
    main()
