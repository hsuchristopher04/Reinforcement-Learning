"""Make a separate comparison viewer; never overwrite the working baseline demo."""
import json
from pathlib import Path
import torch
import cartpole as c
import record_demo as demo

ROOT=Path(__file__).resolve().parent


def main(weighted=False):
    torch.set_num_threads(1)
    experiment=ROOT/('runs/weighted-dagger-v1' if weighted else 'runs/dagger-v1')
    variant='weighted' if weighted else 'aggregated'
    tests=json.loads((experiment/'test_results.json').read_text())
    selection=json.loads((experiment/'selection.json').read_text())['selected']
    output=ROOT/('demo-weighted' if weighted else 'demo-aggregation'); output.mkdir(exist_ok=True)
    teacher=c.load_checkpoint(ROOT/'runs/ablation-mse-seed100/best.pt').eval().requires_grad_(False)
    # Post-hoc illustrative examples; selection does not affect aggregate metrics.
    baseline_returns=tests['baseline']['returns']
    indices=[next(i for i,v in enumerate(baseline_returns) if v==500),
             next(i for i,v in enumerate(baseline_returns) if v<500)]
    test_seed=16_000_000 if weighted else 13_000_000
    seeds=[test_seed+i for i in indices]
    template=(ROOT/'demo_template.html').read_text(encoding='utf-8')
    template=template.replace("model.feature[n]===3?'Angular vel.':'Pole angle'",
                              "['Cart position','Cart velocity','Pole angle','Angular vel.'][model.feature[n]]")
    template=template.replace('The left subtree has a redundant split: both branches choose left. All four original leaves are preserved.',
                              'The full saved policy is preserved, including any redundant branches.')
    template=template.replace('100 episodes','500 episodes').replace('seed 9,000,000',f'seed {test_seed:,}')
    template=template.replace('These are selected success and failure examples, not a random performance sample.',
                              'These examples were selected after testing to illustrate baseline success and failure, not to estimate reliability.')
    for name,model_path in [('baseline',ROOT/'runs/distillation-v1/selected_tree.json'),
                            (variant,experiment/'selected_tree.json')]:
        model=json.loads(model_path.read_text())
        episodes=[]
        for seed in seeds:
            episode={'seed':seed,'teacher':demo.record(teacher,seed),'tree':demo.record(None,seed,model),
                     'label':f'{name.capitalize()} comparison'}
            demo.validate_episode(episode,model)
            episodes.append(episode)
        payload={'env':c.ENV_NAME,'dt':.02,'tree_model':model,'episodes':episodes,
                 'aggregate':{'teacher':tests['teacher'],'tree':tests[name]}}
        serialized=json.dumps(payload,separators=(',',':')).replace('<','\\u003c')
        page=template.replace('__DEMO_DATA__',serialized)
        page=page.replace('02 / Decision tree',f'02 / {name.capitalize()} tree')
        page=page.replace('href="trajectories.json"',f'href="{name}_trajectories.json"')
        (output/f'{name}.html').write_text(page,encoding='utf-8')
        (output/f'{name}_trajectories.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    before,after=tests['baseline'],tests[variant]
    heading='Weighted aggregation: comparison' if weighted else 'Dataset aggregation: before and after'
    conclusion='Weight 0.1 preserved validation completion but did not improve it; further collection was skipped.' if weighted else 'This experiment did not improve the policy.'
    candidate_label='Weighted candidate' if weighted else 'Aggregated tree (worse)'
    report='WEIGHTED_DAGGER.md' if weighted else 'DAGGER.md'
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset aggregation · Before and after</title><style>
body{{margin:0;background:#f5f7f5;color:#192e3b;font:16px/1.5 system-ui}}header{{max-width:1236px;margin:auto;padding:28px 32px 10px}}h1{{margin:0;font-size:28px}}p{{max-width:1000px;color:#526973}}select{{padding:10px;border-radius:8px;font:inherit;border:1px solid #aac4c5}}iframe{{width:100%;height:1350px;border:0}}.note{{background:#fff0e8;padding:12px 16px;border-radius:8px}}a{{color:#087e83}}
</style><header><h1>{heading}</h1>
<p class="note">{conclusion} Original: <b>{before['mean_return']:.2f}/500, {before['completion_rate']:.1%} completion</b>.
{candidate_label}, round {selection['round']}: <b>{after['mean_return']:.2f}/500, {after['completion_rate']:.1%} completion</b>.
All three policies were tested on the same 500 fresh reset seeds. The working baseline remains unchanged.</p>
<label for="variant">Policy to inspect </label><select id="variant"><option value="baseline.html">Original tree</option><option value="{variant}.html">{candidate_label}</option></select>
<p>Switching policies resets playback. The two pages use identical example seeds selected to show baseline success and failure, not to estimate reliability.
<a href="../demo/index.html">Original demo</a> · <a href="../demo-aggregation/index.html">Uniform aggregation demo</a> · <a href="../{report}">Experiment report</a></p></header>
<iframe id="viewer" src="baseline.html" title="Teacher and selected policy playback"></iframe>
<script>document.getElementById('variant').onchange=function(){{document.getElementById('viewer').src=this.value;}};</script></html>'''
    (output/'index.html').write_text(html,encoding='utf-8')
    from presentation import polish_folder
    polish_folder(output)
    print(f'Comparison demo saved: {output / "index.html"}; example seeds {seeds}',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weighted',action='store_true')
    main(parser.parse_args().weighted)
