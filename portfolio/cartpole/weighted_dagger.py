"""Pre-specified weighting comparison, conditional aggregation, and fresh testing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
import cartpole as c
import distill as d
import dagger

ROOT=Path(__file__).resolve().parent
WEIGHTS=[0.0,0.1,0.25,0.5,1.0]


def fit_weighted(x,y,new_x,new_y,weight):
    all_x=np.concatenate([x,new_x]); all_y=np.concatenate([y,new_y])
    weights=np.r_[np.ones(len(y)),np.full(len(new_y),weight)]
    tree=DecisionTreeClassifier(max_depth=2,random_state=2026).fit(all_x,all_y,sample_weight=weights)
    model=d.export_model(tree)
    assert model['leaves']<=4
    probes=all_x[::max(1,len(all_x)//2000)]
    np.testing.assert_array_equal(tree.predict(probes),[d.tree_action(model,s) for s in probes])
    return tree,model


def evaluate_model(model,episodes,seed):
    env=c.gym.make(c.ENV_NAME); returns=[]; counts={'position_failures':0,'angle_failures':0,'completed':0}
    try:
        for episode in range(episodes):
            obs,_=env.reset(seed=seed+episode); total=0
            while True:
                obs,reward,terminated,truncated,_=env.step(d.tree_action(model,obs)); total+=reward
                if terminated or truncated:
                    counts['position_failures']+=int(abs(obs[0])>env.unwrapped.x_threshold)
                    counts['angle_failures']+=int(abs(obs[2])>env.unwrapped.theta_threshold_radians)
                    counts['completed']+=int(truncated and not terminated)
                    break
            returns.append(total)
    finally: env.close()
    return dict(episodes=episodes,seed=seed,mean_return=float(np.mean(returns)),std_return=float(np.std(returns)),
                min_return=float(min(returns)),completion_rate=counts['completed']/episodes,returns=returns,**counts)


def choose_weight(rows):
    return max([r for r in rows if r['weight']>0],key=lambda r:(r['completion_rate'],r['mean_return'],-r['weight']))


def run(output):
    if output.exists(): raise SystemExit(f'Already exists: {output}; choose another --output directory')
    output.mkdir(parents=True)
    source=output/'source';source.mkdir()
    for name in ['weighted_dagger.py','dagger.py','distill.py','cartpole.py','model.py','utils.py']:
        (source/name).write_bytes((ROOT/name).read_bytes())
    original=ROOT/'runs/distillation-v1/demonstrations.npz'
    batch_path=ROOT/'runs/dagger-v1/round1_new_examples.npz'
    baseline_path=ROOT/'runs/distillation-v1/selected_tree.json'
    uniform_path=ROOT/'runs/dagger-v1/round1_tree.json'
    teacher_path=ROOT/'runs/ablation-mse-seed100/best.pt'
    inputs=[original,batch_path,baseline_path,uniform_path,teacher_path]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    d.write_json(output/'config.json',dict(weights=WEIGHTS,max_depth=2,tree_seed=2026,validation_seed=14_000_000,
        validation_episodes=200,test_seed=16_000_000,test_episodes=500,
        continuation_seeds=[15_000_000+r*1000 for r in range(2,6)],
        continuation_gate='strict improvement in validation completion over baseline; otherwise stop',
        selection='completion, then mean, then smaller weight; best continuation round by completion then mean then earliest',
        weighting='original samples 1.0; all accumulated new samples fixed selected weight',
        input_sha256=hashes,sklearn=sklearn.__version__,numpy=np.__version__,torch=str(torch.__version__),gymnasium=c.gym.__version__))
    with np.load(original,allow_pickle=False) as data:
        mask=np.isin(data['episode_ids'],data['train_episode_ids'])
        x,y=data['observations'][mask],data['actions'][mask]
        assert len(y)==80000 and not set(data['train_episode_ids'])&set(data['validation_episode_ids'])
    with np.load(batch_path,allow_pickle=False) as data: new_x,new_y=data['observations'].copy(),data['teacher_actions'].copy()
    assert len(new_y)==24295
    baseline=json.loads(baseline_path.read_text()); uniform=json.loads(uniform_path.read_text())
    baseline_result=evaluate_model(baseline,200,14_000_000)
    d.write_json(output/'baseline_validation.json',baseline_result)
    print(f'Baseline validation: {baseline_result["mean_return"]:.2f}, {baseline_result["completion_rate"]:.1%}',flush=True)
    rows=[]; models={}
    for weight in WEIGHTS:
        tree,model=fit_weighted(x,y,new_x,new_y,weight)
        if weight==0: assert model==baseline,'Weight-zero control differs from baseline'
        if weight==1: assert model==uniform,'Weight-one control differs from previous first round'
        tag=str(weight).replace('.','p')
        d.write_json(output/f'weight_{tag}_tree.json',model)
        (output/f'weight_{tag}_rules.txt').write_text(export_text(tree,feature_names=d.FEATURES,decimals=8),encoding='utf-8')
        result=evaluate_model(model,200,14_000_000)
        d.write_json(output/f'weight_{tag}_validation.json',result)
        row=dict(weight=weight,round=1,leaves=model['leaves'],**{k:result[k] for k in ['mean_return','min_return','completion_rate','position_failures','angle_failures']})
        rows.append(row);models[weight]=model
        print(f'Weight {weight}: {row}',flush=True)
    d.write_json(output/'weight_comparison.json',rows)
    d.write_json(output/'control_checks.json',dict(zero_exactly_matches_baseline=True,one_exactly_matches_prior_round1=True))
    best=choose_weight(rows); selected=best.copy(); chosen_model=models[best['weight']]
    continue_rounds=best['completion_rate']>baseline_result['completion_rate']
    d.write_json(output/'continuation_decision.json',dict(continue_rounds=continue_rounds,chosen_weight=best['weight'],
        baseline_completion=baseline_result['completion_rate'],candidate_completion=best['completion_rate']))
    teacher=c.load_checkpoint(teacher_path).eval().requires_grad_(False)
    round_rows=[best]; current=chosen_model
    if continue_rounds:
        for round_number in range(2,6):
            batch,outcomes=dagger.collect_labeled(teacher,current,50,15_000_000+round_number*1000)
            np.savez_compressed(output/f'round{round_number}_new_examples.npz',**batch)
            d.write_json(output/f'round{round_number}_collection.json',outcomes)
            new_x=np.concatenate([new_x,batch['observations']]);new_y=np.concatenate([new_y,batch['teacher_actions']])
            tree,current=fit_weighted(x,y,new_x,new_y,best['weight'])
            d.write_json(output/f'round{round_number}_tree.json',current)
            (output/f'round{round_number}_rules.txt').write_text(export_text(tree,feature_names=d.FEATURES,decimals=8),encoding='utf-8')
            result=evaluate_model(current,200,14_000_000)
            d.write_json(output/f'round{round_number}_validation.json',result)
            row=dict(weight=best['weight'],round=round_number,leaves=current['leaves'],new_examples_total=len(new_y),
                     **{k:result[k] for k in ['mean_return','min_return','completion_rate','position_failures','angle_failures']})
            round_rows.append(row)
            if (row['completion_rate'],row['mean_return'])>(selected['completion_rate'],selected['mean_return']): selected,chosen_model=row,current
            print(f'Round {round_number}: {row}',flush=True)
    else: print('No validation completion improvement: further collection skipped as specified.',flush=True)
    d.write_json(output/'rounds.json',round_rows)
    d.write_json(output/'selection.json',dict(selected=selected,continued=continue_rounds,selected_before_test=True))
    d.write_json(output/'selected_tree.json',chosen_model)
    tests={}
    for name,model in [('baseline',baseline),('weighted',chosen_model)]:
        tests[name]=evaluate_model(model,500,16_000_000)
        print(f'Test {name}: mean {tests[name]["mean_return"]:.2f}, completion {tests[name]["completion_rate"]:.1%}',flush=True)
    tests['teacher']=d.rollout(lambda obs:c.select_action(teacher,obs),500,16_000_000)
    d.write_json(output/'test_results.json',tests)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(WEIGHTS,[r['completion_rate'] for r in rows],'o-',color='#087e83')
    ax.axhline(baseline_result['completion_rate'],linestyle='--',color='gray',label='Baseline')
    ax.set(xlabel='Weight per new example (original = 1.0)',ylabel='Validation completion rate',ylim=(0,1.05),title='Fixed data and tree size: weighting comparison');ax.legend();fig.tight_layout();fig.savefig(output/'weight_comparison.png',dpi=160);plt.close(fig)
    lines=['# Weighted aggregation experiment','','Both controls reproduced their reference trees exactly.','',
           '| Weight | Validation mean | Completion | Position failures | Angle failures |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['weight']} | {r['mean_return']:.2f} | {r['completion_rate']:.1%} | {r['position_failures']} | {r['angle_failures']} |" for r in rows]
    lines += ['',f"Continued aggregation: {continue_rounds}. Selected weight {selected['weight']}, round {selected['round']}.",
              '', '| Policy | Test mean | Completion | Minimum |','|---|---:|---:|---:|']
    lines += [f"| {name} | {r['mean_return']:.2f} | {r['completion_rate']:.1%} | {r['min_return']:.0f} |" for name,r in tests.items()]
    lines += ['','![Weight comparison](weight_comparison.png)','','500 matched fresh test seeds; selection completed before test evaluation. One fitting seed and dataset. Original validation examples were excluded from fitting. No post-test tuning occurred.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Complete: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'runs/weighted-dagger-v1')
    args=parser.parse_args();torch.set_num_threads(1);run(args.output)
