"""Build a separate capacity comparison with arbitrary saved tree layouts."""
import json
import re
from pathlib import Path
import torch
import cartpole as c
import record_demo as demo

ROOT=Path(__file__).resolve().parent


def fit_capacity_tree(page):
    """Only the eight-leaf page uses a full-width, responsive tree diagram."""
    page=page.replace('<div style="overflow:auto;max-height:560px"><svg id="treeDiagram"',
                      '<div style="min-width:0"><svg id="treeDiagram"')
    page=page.replace('`width:${diagramWidth}px;max-width:none;height:${diagramHeight}px`',
                      '`width:100%;max-width:100%;height:auto;display:block`')
    page=page.replace('</style>', '.rulelayout{grid-template-columns:minmax(0,1fr)}'
                      '.rulelayout .obs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:24px}'
                      '@media(max-width:500px){.rulelayout .obs{grid-template-columns:1fr}}</style>',1)
    return page.replace('All saved nodes are shown. Scroll the diagram horizontally to inspect larger trees.',
                        'All eight leaves are shown together. The diagram scales to fit the available width.')


def main():
    torch.set_num_threads(1)
    experiment=ROOT/'runs/capacity-v1';output=ROOT/'demo-capacity';output.mkdir(exist_ok=True)
    results=json.loads((experiment/'test_results.json').read_text())
    baseline_returns=results['baseline']['returns'];candidate_returns=results['capacity']['returns']
    criteria=[('Larger tree succeeds; baseline fails',lambda b,n:b<500 and n==500),
              ('Both trees complete',lambda b,n:b==500 and n==500),
              ('Larger tree still fails',lambda b,n:n<500)]
    examples=[(label,next(i for i,(b,n) in enumerate(zip(baseline_returns,candidate_returns)) if predicate(b,n)))
              for label,predicate in criteria]
    teacher=c.load_checkpoint(ROOT/'runs/ablation-mse-seed100/best.pt').eval().requires_grad_(False)
    template=(ROOT/'demo_template.html').read_text(encoding='utf-8')
    template=template.replace('a four-leaf decision tree','a saved decision tree').replace('Four-leaf tree','Selected tree')
    template=template.replace("model.feature[n]===3?'Angular vel.':'Pole angle'", "['Cart position','Cart velocity','Pole angle','Angular vel.'][model.feature[n]]")
    template=template.replace('The left subtree has a redundant split: both branches choose left. All four original leaves are preserved.',
                              'All saved nodes are shown. Scroll the diagram horizontally to inspect larger trees.')
    template=template.replace('100 episodes','500 episodes').replace('seed 9,000,000','seed 18,000,000')
    template=template.replace('These are selected success and failure examples, not a random performance sample.',
                              'These are deliberately selected rescue, shared-success, and remaining-failure examples, not a random sample.')
    template=template.replace('<svg id="treeDiagram"','<div style="overflow:auto;max-height:560px"><svg id="treeDiagram"')
    template=template.replace('</svg><div><div class="obs"','</svg></div><div><div class="obs"')
    layout="""const model=data.tree_model,positions={};
let leafIndex=0,deepest=0;
function layoutNode(n,depth){deepest=Math.max(deepest,depth);let x;
if(model.children_left[n]===-1){x=90+170*leafIndex++;}
else{x=(layoutNode(model.children_left[n],depth+1)+layoutNode(model.children_right[n],depth+1))/2;}
positions[n]=[x,35+95*depth];return x;}
layoutNode(0,0);const diagramWidth=Math.max(700,leafIndex*170+10),diagramHeight=95*deepest+75;
$('treeDiagram').setAttribute('viewBox',`0 0 ${diagramWidth} ${diagramHeight}`);
$('treeDiagram').setAttribute('style',`width:${diagramWidth}px;max-width:none;height:${diagramHeight}px`);"""
    template,count=re.subn(r'const model=data\.tree_model,positions=\{[^\n]+\};',lambda _:layout,template)
    assert count==1,'Tree layout placeholder not found'
    for name,file,label in [('baseline',ROOT/'runs/distillation-v1/selected_tree.json','Original · 4 leaves'),
                            ('capacity',experiment/'selected_tree.json','Selected · 8 leaves')]:
        model=json.loads(file.read_text());episodes=[]
        for example,i in examples:
            seed=18_000_000+i
            episode=dict(seed=seed,label=example,teacher=demo.record(teacher,seed),tree=demo.record(None,seed,model))
            demo.validate_episode(episode,model);episodes.append(episode)
        data=dict(env=c.ENV_NAME,dt=.02,tree_model=model,episodes=episodes,
                  aggregate=dict(teacher=results['teacher'],tree=results[name]))
        page=template.replace('__DEMO_DATA__',json.dumps(data,separators=(',',':')).replace('<','\\u003c'))
        page=page.replace('02 / Decision tree',f'02 / {label}').replace('href="trajectories.json"',f'href="{name}_trajectories.json"')
        if name=='capacity':
            page=fit_capacity_tree(page)
        (output/f'{name}.html').write_text(page,encoding='utf-8')
        (output/f'{name}_trajectories.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Four leaves or eight? · CartPole</title><style>body{margin:0;background:#f5f7f5;color:#192e3b;font:16px/1.5 system-ui}header{max-width:1236px;margin:auto;padding:24px 32px 0}h1{margin:0}p{max-width:1100px;color:#526973}select{padding:10px;border-radius:8px;font:inherit;border:1px solid #aac4c5}iframe{width:100%;height:1450px;border:0}a{color:#087e83}.result{background:#dff2e9;padding:14px;border-radius:10px}</style>
<header><h1>Four leaves or eight?</h1><p class="result">On 500 matched fresh episodes: original four-leaf tree <b>485.66 return / 70.8% completion</b>; selected eight-leaf tree <b>498.84 / 98.4%</b>. Teacher: <b>500 / 100%</b>.</p>
<p>The eight-leaf policy was the smallest candidate meeting 90% validation completion. It still failed in 8 of 500 test episodes; its shortest episode lasted 288 steps. Examples were selected after testing to illustrate both gains and limitations.</p>
<label for="variant">Policy to inspect </label><select id="variant"><option value="baseline.html">Original tree · 4 leaves</option><option value="capacity.html" selected>Selected tree · 8 leaves</option></select>
<p>Switching resets playback. Both pages use the same example seeds. The eight-leaf tree fits in full without diagram scrolling. <a href="../CAPACITY.md">Experiment report</a> · <a href="../demo/index.html">Original demo</a></p></header>
<iframe id="viewer" src="capacity.html" title="Teacher and tree policy playback"></iframe><script>document.getElementById('variant').onchange=function(){document.getElementById('viewer').src=this.value;};</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    from presentation import polish_folder
    polish_folder(output)
    print(f'Demo built: {output}; examples {[(label,18000000+i) for label,i in examples]}',flush=True)


if __name__=='__main__':main()
