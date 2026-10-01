"""Fixed eight-leaf replication: five datasets, three fitting seeds, no selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sklearn
import torch
from sklearn.tree import DecisionTreeClassifier, export_text
import cartpole as c
import distill as d
from weighted_dagger import evaluate_model

ROOT=Path(__file__).resolve().parent
COLLECTION_SEEDS=[20_000_000+i*1000 for i in range(5)]
FIT_SEEDS=[2026,2027,2028]
EVAL_SEED=21_000_000


def fingerprint(model):
    return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def summarize(rows):
    groups=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        items=[r for r in rows if r['dataset']==dataset]
        groups.append(dict(dataset=dataset,mean_completion=float(np.mean([r['completion_rate'] for r in items])),
            min_completion=min(r['completion_rate'] for r in items),max_completion=max(r['completion_rate'] for r in items),
            mean_return=float(np.mean([r['mean_return'] for r in items])),distinct_trees=len({r['fingerprint'] for r in items})))
    return dict(datasets=groups,mean_dataset_completion=float(np.mean([g['mean_completion'] for g in groups])),
                min_dataset_completion=min(g['mean_completion'] for g in groups),
                max_dataset_completion=max(g['mean_completion'] for g in groups),
                distinct_trees=len({r['fingerprint'] for r in rows}))


def run(output):
    if output.exists(): raise SystemExit(f'Output exists: {output}; use a new --output folder')
    output.mkdir(parents=True);source=output/'source';source.mkdir()
    for name in ['replication.py','distill.py','weighted_dagger.py','dagger.py','cartpole.py','model.py','utils.py']:
        (source/name).write_bytes((ROOT/name).read_bytes())
    teacher_path=ROOT/'runs/ablation-mse-seed100/best.pt'; reference_path=ROOT/'runs/capacity-v1/selected_tree.json'
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [teacher_path,reference_path]}
    d.write_json(output/'config.json',dict(collection_seeds=COLLECTION_SEEDS,episodes_per_dataset=200,
        train_episodes=160,validation_episodes=40,split_seed=2026,fit_seeds=FIT_SEEDS,max_leaf_nodes=8,
        max_depth=None,evaluation_seed=EVAL_SEED,evaluation_episodes=500,selection='none; report all 15 fits',
        exact_duplicate_policy_evaluation='reuse identical deterministic policy results on identical seeds',
        input_sha256=hashes,sklearn=sklearn.__version__,numpy=np.__version__,torch=str(torch.__version__),gymnasium=c.gym.__version__))
    teacher=c.load_checkpoint(teacher_path).eval().requires_grad_(False)
    rows=[];cache={}
    for dataset,seed in enumerate(COLLECTION_SEEDS,1):
        print(f'Dataset {dataset}/5; collection seed {seed}',flush=True)
        folder=output/f'dataset{dataset}';folder.mkdir()
        data=d.collect(teacher,200,seed)
        train,val=d.episode_split(data['episode_ids'],2026)
        assert len(train)==160 and len(val)==40 and not set(train)&set(val)
        data.update(train_episode_ids=train,validation_episode_ids=val)
        np.savez_compressed(folder/'demonstrations.npz',**data)
        train_mask=np.isin(data['episode_ids'],train);val_mask=np.isin(data['episode_ids'],val)
        x,y=data['observations'][train_mask],data['actions'][train_mask]
        for fit_seed in FIT_SEEDS:
            tree=DecisionTreeClassifier(max_leaf_nodes=8,random_state=fit_seed).fit(x,y)
            model=d.export_model(tree);assert model['leaves']<=8
            probes=data['observations'][::50]
            np.testing.assert_array_equal(tree.predict(probes),[d.tree_action(model,s) for s in probes])
            digest=fingerprint(model)
            d.write_json(folder/f'tree_seed{fit_seed}.json',model)
            (folder/f'rules_seed{fit_seed}.txt').write_text(export_text(tree,feature_names=d.FEATURES,max_depth=100,decimals=8),encoding='utf-8')
            reused=digest in cache
            if not reused: cache[digest]=evaluate_model(model,500,EVAL_SEED)
            result=cache[digest]
            d.write_json(folder/f'evaluation_seed{fit_seed}.json',dict(**result,reused_identical_policy=reused))
            row=dict(dataset=dataset,fit_seed=fit_seed,leaves=model['leaves'],depth=model['depth'],fingerprint=digest,
                training_examples=len(y),validation_examples=int(val_mask.sum()),
                action_agreement=float(tree.score(data['observations'][val_mask],data['actions'][val_mask])),
                **{k:result[k] for k in ['mean_return','completion_rate','position_failures','angle_failures']})
            rows.append(row);d.write_json(output/'all_results.json',rows)
            print(f'Dataset {dataset}, fit seed {fit_seed}: completion {row["completion_rate"]:.1%}, mean {row["mean_return"]:.2f}; identical cached policy: {reused}',flush=True)
    reference=json.loads(reference_path.read_text())
    references={'saved_eight_leaf':evaluate_model(reference,500,EVAL_SEED),
                'teacher':d.rollout(lambda obs:c.select_action(teacher,obs),500,EVAL_SEED)}
    d.write_json(output/'references.json',references)
    summary=summarize(rows);d.write_json(output/'summary.json',summary)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for j,fit_seed in enumerate(FIT_SEEDS):
        items=[r for r in rows if r['fit_seed']==fit_seed]
        ax.scatter([r['dataset']+(j-1)*.12 for r in items],[r['completion_rate']*100 for r in items],label=f'Fitting seed {fit_seed}',s=60)
    ax.axhline(references['saved_eight_leaf']['completion_rate']*100,color='black',linestyle='--',label='Saved eight-leaf reference')
    ax.set(xlabel='Independent demonstration dataset',ylabel='Completion rate (%)',xticks=range(1,6),ylim=(-3,105),title='Eight-leaf policy replication: all 15 fits');ax.legend();fig.tight_layout();fig.savefig(output/'replication.png',dpi=160);plt.close(fig)
    lines=['# Eight-leaf replication: all results','','| Dataset | Fitting seed | Leaves | Depth | Mean return | Completion | Track failures | Pole failures |','|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['fit_seed']} | {r['leaves']} | {r['depth']} | {r['mean_return']:.2f} | {r['completion_rate']:.1%} | {r['position_failures']} | {r['angle_failures']} |" for r in rows]
    lines += ['','## Dataset-level summary','','| Dataset | Mean completion across fitting seeds | Range | Distinct trees |','|---|---:|---:|---:|']
    lines += [f"| {g['dataset']} | {g['mean_completion']:.1%} | {g['min_completion']:.1%}–{g['max_completion']:.1%} | {g['distinct_trees']} |" for g in summary['datasets']]
    lines += ['',f"Mean across the five dataset means: {summary['mean_dataset_completion']:.1%}; range {summary['min_dataset_completion']:.1%}–{summary['max_dataset_completion']:.1%}. {summary['distinct_trees']} distinct serialized policies among 15 fits.",
        '',f"Saved eight-leaf reference on the same 500 seeds: {references['saved_eight_leaf']['completion_rate']:.1%} completion, {references['saved_eight_leaf']['mean_return']:.2f} mean return.",
        f"Teacher: {references['teacher']['completion_rate']:.1%} completion, {references['teacher']['mean_return']:.2f} mean return.",
        '','![Replication results](replication.png)','','## Interpretation limits','',
        'Five independent demonstration collections are the main replication units. Three fitting seeds within each dataset are not independent datasets; exact duplicate trees are not extra evidence.',
        'All policies share 500 reset seeds, making evaluations paired. Episode outcomes across policies are not independent observations. No winning tree was selected and no tuning followed evaluation.',
        'The frozen teacher, environment distribution, sample size, and fitting method are unchanged. This does not test changed physics, broader resets, or other teachers.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'runs/replication-v1')
    args=parser.parse_args();torch.set_num_threads(1);run(args.output)
