"""Copy curated models and evidence out of Git-ignored runs, without changing runs."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNS=['ablation-mse-seed100','distillation-v1','dagger-v1','weighted-dagger-v1','capacity-v1','replication-v1','comparison','mse-seed-comparison']


def main():
    target=ROOT/'artifacts';(target/'models').mkdir(parents=True,exist_ok=True)
    entries=[]
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        entries.append(dict(path=str(destination.relative_to(ROOT)).replace('\\','/'),
                            source=str(source.relative_to(ROOT)).replace('\\','/'),
                            bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    for name,source in [('teacher.pt','ablation-mse-seed100/best.pt'),
                        ('tree4.json','distillation-v1/selected_tree.json'),('tree8.json','capacity-v1/selected_tree.json')]:
        copy(ROOT/'runs'/source,target/'models'/name)
    for run in RUNS:
        for source in sorted((ROOT/'runs'/run).glob('*')):
            if source.is_file() and source.suffix.lower() in ['.json','.csv','.png','.txt','.md']:
                copy(source,target/'results'/run/source.name)
    (target/'manifest.json').write_text(json.dumps(dict(format_version=1,
        description='Curated exact copies; source paths identify local experiment provenance.',files=entries),indent=2),encoding='utf-8')
    print(f'Packaged {len(entries)} files, {sum(e["bytes"] for e in entries):,} bytes. Original runs unchanged.')


if __name__=='__main__':main()
