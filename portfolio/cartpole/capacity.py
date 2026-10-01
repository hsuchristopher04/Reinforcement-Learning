"""Fixed-data leaf-budget experiment: ten candidates, validation-first selection."""
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
BUDGETS=[4,8,16,32,64]


def select_candidate(rows):
    qualified=[r for r in rows if r['completion_rate']>=.9]
    if qualified:
        return min(qualified,key=lambda r:(r['leaves'],-r['completion_rate'],-r['mean_return'],r['dataset'])),True
    return min(rows,key=lambda r:(-r['completion_rate'],-r['mean_return'],r['leaves'],r['dataset'])),False


def run(output):
    if output.exists(): raise SystemExit(f'Already exists: {output}; choose a new --output')
    output.mkdir(parents=True); source=output/'source';source.mkdir()
    for name in ['capacity.py','weighted_dagger.py','dagger.py','distill.py','cartpole.py','model.py','utils.py']:
        (source/name).write_bytes((ROOT/name).read_bytes())
    original=ROOT/'runs/distillation-v1/demonstrations.npz'
    batch=ROOT/'runs/dagger-v1/round1_new_examples.npz'
    basefile=ROOT/'runs/distillation-v1/selected_tree.json'
    teacherfile=ROOT/'runs/ablation-mse-seed100/best.pt'
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [original,batch,basefile,teacherfile]}
    d.write_json(output/'config.json',dict(budgets=BUDGETS,datasets=['original','augmented'],tree_seed=2026,
        max_depth=None,criterion='gini',sample_weight='uniform',validation_seed=17_000_000,validation_episodes=200,
        test_seed=18_000_000,test_episodes=500,selection='smallest actual leaves with >=90% completion; ties completion then return; fallback highest completion then return',
        input_sha256=hashes,sklearn=sklearn.__version__,numpy=np.__version__,torch=str(torch.__version__),gymnasium=c.gym.__version__))
    with np.load(original,allow_pickle=False) as data:
        assert not set(data['train_episode_ids'])&set(data['validation_episode_ids'])
        mask=np.isin(data['episode_ids'],data['train_episode_ids']);x=data['observations'][mask];y=data['actions'][mask]
        assert len(y)==80000
    with np.load(batch,allow_pickle=False) as data: ax=np.r_[x,data['observations']];ay=np.r_[y,data['teacher_actions']]
    assert len(ay)==104295
    baseline=json.loads(basefile.read_text())
    base_validation=evaluate_model(baseline,200,17_000_000)
    d.write_json(output/'baseline_validation.json',base_validation)
    rows=[]
    for dataset,features,labels in [('original',x,y),('augmented',ax,ay)]:
        for budget in BUDGETS:
            tree=DecisionTreeClassifier(max_leaf_nodes=budget,random_state=2026).fit(features,labels)
            model=d.export_model(tree); assert model['leaves']<=budget
            probes=features[::max(1,len(features)//2000)]
            np.testing.assert_array_equal(tree.predict(probes),[d.tree_action(model,s) for s in probes])
            tag=f'{dataset}_{budget}'
            d.write_json(output/f'{tag}_tree.json',model)
            (output/f'{tag}_rules.txt').write_text(export_text(tree,feature_names=d.FEATURES,max_depth=100,decimals=8),encoding='utf-8')
            result=evaluate_model(model,200,17_000_000)
            d.write_json(output/f'{tag}_validation.json',result)
            row=dict(dataset=dataset,budget=budget,leaves=model['leaves'],depth=model['depth'],
                     **{k:result[k] for k in ['mean_return','completion_rate','position_failures','angle_failures']})
            rows.append(row);d.write_json(output/'comparison.json',rows);print(row,flush=True)
    selected,qualified=select_candidate(rows);tag=f"{selected['dataset']}_{selected['budget']}"
    d.write_json(output/'selection.json',dict(selected=selected,qualified=qualified,selected_before_test=True))
    (output/'selected_tree.json').write_bytes((output/f'{tag}_tree.json').read_bytes())
    (output/'selected_rules.txt').write_bytes((output/f'{tag}_rules.txt').read_bytes())
    model=json.loads((output/'selected_tree.json').read_text())
    tests={}
    for name,policy in [('baseline',baseline),('capacity',model)]:
        tests[name]=evaluate_model(policy,500,18_000_000)
        print(f"Test {name}: {tests[name]['mean_return']:.2f}, completion {tests[name]['completion_rate']:.1%}",flush=True)
    teacher=c.load_checkpoint(teacherfile).eval().requires_grad_(False)
    tests['teacher']=d.rollout(lambda obs:c.select_action(teacher,obs),500,18_000_000)
    d.write_json(output/'test_results.json',tests)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for dataset,color in [('original','#087e83'),('augmented','#b55d21')]:
        points=[r for r in rows if r['dataset']==dataset]
        ax.plot([r['leaves'] for r in points],[r['completion_rate'] for r in points],'o-',label=dataset,color=color)
    ax.axhline(.9,linestyle='--',color='gray',label='90% selection target')
    ax.axhline(base_validation['completion_rate'],linestyle=':',color='black',label='Saved baseline')
    ax.set(xscale='log',xlabel='Actual number of leaves',ylabel='Validation completion rate',ylim=(0,1.05),title='Tree capacity × training dataset');ax.legend();fig.tight_layout();fig.savefig(output/'capacity_comparison.png',dpi=160);plt.close(fig)
    lines=['# Leaf-budget comparison','','| Dataset | Budget | Actual leaves | Depth | Validation mean | Completion | Position failures | Angle failures |','|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['budget']} | {r['leaves']} | {r['depth']} | {r['mean_return']:.2f} | {r['completion_rate']:.1%} | {r['position_failures']} | {r['angle_failures']} |" for r in rows]
    lines += ['',f"Selected {tag}; validation target met: {qualified}. Saved baseline validation completion: {base_validation['completion_rate']:.1%}.",'','## Fresh test (500 matched episodes)','','| Policy | Mean | Std | Minimum | Completion |','|---|---:|---:|---:|---:|']
    lines += [f"| {name} | {r['mean_return']:.2f} | {r['std_return']:.2f} | {r['min_return']:.0f} | {r['completion_rate']:.1%} |" for name,r in tests.items()]
    lines += ['','![Capacity comparison](capacity_comparison.png)','','One dataset pair and fitting seed. New validation and test reset seeds. No post-test tuning. max_leaf_nodes uses best-first growth and differs from the previous max_depth cap. Original action-validation examples excluded from fitting.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('Complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'runs/capacity-v1')
    args=parser.parse_args();torch.set_num_threads(1);run(args.output)
