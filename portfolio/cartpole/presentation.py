"""Apply consistent presentation fixes to recorded demos without changing data."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DOCS = 'https://github.com/hsuchristopher04/Reinforcement-Learning/blob/main/portfolio/cartpole/'
CSS = '''/* responsive-review */
select{max-width:100%;min-width:0}#examples{width:min(100%,420px);max-width:100%}
.controls>*{max-width:100%}.panelhead{gap:10px;flex-wrap:wrap}
.callout{white-space:pre-line}footer{gap:12px;flex-wrap:wrap}
@media(max-width:480px){.stats{grid-template-columns:1fr}.stat+.stat{border-left:0;border-top:1px solid var(--line)}.timeline{flex-wrap:wrap}#clock{width:100%;text-align:right}.controls #examples{width:100%}.readouts{padding:12px}.rules{padding:14px}.obs div{gap:12px}.obs b{white-space:nowrap}}
'''
RESIZE = '''<script data-frame-resize>
(()=>{const frame=document.getElementById('viewer');let observer;
function fit(){try{const body=frame.contentDocument.body;frame.style.height=Math.ceil(body.getBoundingClientRect().height+32)+'px';}catch{}}
frame.addEventListener('load',()=>{if(observer)observer.disconnect();try{observer=new ResizeObserver(fit);observer.observe(frame.contentDocument.body);fit();}catch{}});
window.addEventListener('resize',fit);fit();})();
</script>'''


def polish_page(page):
    if 'responsive-review' not in page:
        page = page.replace('</style>', CSS + '</style>', 1)
    page = page.replace('Math.round(m.completion_rate*100)', 'Number((m.completion_rate*100).toFixed(1))')
    # Preserve diagrams, adding a legible version of the same recorded path.
    needle = 'reveals the next state.`;}'
    replacement = '''reveals the next state.`;
if(f.action!==null){const rules=f.path.filter(n=>model.children_left[n]!==-1).map(n=>{const next=f.path[f.path.indexOf(n)+1];return `${names[model.feature[n]]} ${next===model.children_left[n]?'≤':'>'} ${model.threshold[n].toFixed(4)} ${units[model.feature[n]]}`;});$('explanation').textContent+='\\nActive rules (rounded):\\n'+rules.join('\\n');}}'''
    page = page.replace(needle, replacement)
    if 'id="viewer"' in page and 'data-frame-resize' not in page:
        page = page.replace('</html>', RESIZE + '</html>')
    # GitHub Pages may render .md as .html rather than serving .md links.
    page = re.sub(r'href="(?:\.\./)*([A-Z_]+\.md)"', lambda m: 'href="'+DOCS+m[1]+'"', page)
    if 'Back to project' not in page:
        page = page.replace('<header>', '<header><nav><a href="../index.html">← Back to project</a></nav>', 1) if 'id="viewer"' in page else page
    return page


def polish_folder(folder):
    for path in Path(folder).glob('*.html'):
        path.write_text(polish_page(path.read_text(encoding='utf-8')), encoding='utf-8')


if __name__ == '__main__':
    for name in ['demo','demo-aggregation','demo-weighted','demo-capacity']:
        polish_folder(ROOT/name)
    path = ROOT/'demo_template.html'
    path.write_text(polish_page(path.read_text(encoding='utf-8')), encoding='utf-8')
    path = ROOT/'index.html'
    page = re.sub(r'href="([A-Z_]+\.md)"', lambda m: 'href="'+DOCS+m[1]+'"', path.read_text(encoding='utf-8'))
    page = page.replace('Documentation links open Markdown source in a local browser.', 'Documentation links open the formatted reports on GitHub.')
    path.write_text(page,encoding='utf-8')
