"""Single entry point for packaged policies, training, and distillation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def verify():
    manifest=json.loads((ROOT/'artifacts/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=(ROOT/item['path']).resolve()
        if not path.is_relative_to((ROOT/'artifacts').resolve()):
            raise SystemExit('Artifact path must stay inside artifacts/')
        if not path.is_file() or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise SystemExit(f'Artifact missing or changed: {path}')
    print(f"Verified {len(manifest['files'])} packaged files.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Verify bundled model and evidence hashes (standard library only)')
    sub.add_parser('demo',help='Print the local demo landing-page path')
    evaluation=sub.add_parser('evaluate',help='Evaluate bundled policies; no runs folder needed')
    evaluation.add_argument('policy',choices=['teacher','tree4','tree8','all'],default='all',nargs='?')
    evaluation.add_argument('--episodes',type=int,default=100)
    evaluation.add_argument('--seed',type=int,default=22_000_000)
    sub.add_parser('train',help='Forward remaining arguments to cartpole.py train')
    sub.add_parser('distill',help='Forward remaining arguments to distill.py run, using the bundled teacher')
    args,extra=parser.parse_known_args()
    if args.command in ['train','distill']:
        script,command=('cartpole.py','train') if args.command=='train' else ('distill.py','run')
        defaults=[] if args.command=='train' else ['--teacher',str(ROOT/'artifacts/models/teacher.pt')]
        raise SystemExit(subprocess.call([sys.executable,str(ROOT/script),command,*defaults,*extra],cwd=ROOT))
    if extra:parser.error(f'Unrecognized arguments: {extra}')
    if args.command=='verify':verify();return
    if args.command=='demo':print(ROOT/'index.html');return
    if args.episodes<=0 or args.seed<0:parser.error('Episodes must be positive and seed nonnegative')
    verify()
    import torch
    import cartpole as c
    import distill as d
    torch.set_num_threads(1)
    for name in (['teacher','tree4','tree8'] if args.policy=='all' else [args.policy]):
        if name=='teacher':
            policy=c.load_checkpoint(ROOT/'artifacts/models/teacher.pt').eval()
            result=d.rollout(lambda obs:c.select_action(policy,obs),args.episodes,args.seed)
        else:
            model=json.loads((ROOT/f'artifacts/models/{name}.json').read_text())
            result=d.rollout(lambda obs:d.tree_action(model,obs),args.episodes,args.seed)
        print(json.dumps(dict(policy=name,**{k:v for k,v in result.items() if k!='returns'}),indent=2),flush=True)


if __name__=='__main__':main()
