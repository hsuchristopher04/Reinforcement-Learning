// Browser-independent unit tests of playback logic using a minimal fake DOM.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, 'demo/index.html'), 'utf8');
const payload = html.match(/<script id="demo-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element() {
  const classes = new Set();
  return {children: [], value: '0', textContent: '', width: 1000, height: 420,
    classList: {add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c)},
    setAttribute(k, v) { this[k] = v; if (k === 'id') elements.set(v, this); },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    getContext() { return new Proxy({}, {get: () => () => {}}); }};
}
const get = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
get('demo-data').textContent = payload;
get('speed').value = '1';
let frame;
const context = vm.createContext({document: {getElementById: get,
  createElement: element, createElementNS: element, addEventListener() {},
  querySelectorAll: () => [...elements.values()].filter(e => e.class === 'node' || e.class === 'edge')},
  requestAnimationFrame: callback => {frame = callback;}, console});
vm.runInContext(script, context);
assert.equal(get('treeMean').textContent, '486.93 / 500');
get('step').onclick();
assert.equal(get('teacherSteps').textContent, '1 / 500');
get('examples').value = '1'; get('examples').onchange();
get('seek').value = '439'; get('seek').oninput();
assert.equal(get('treeStatus').textContent, 'Failed');
assert.equal(get('teacherStatus').textContent, 'Running');
assert.equal(get('treeAction').textContent, '—');
assert(!get('node-0').classList.contains('active'));
get('seek').value = '450'; get('seek').oninput();
assert.equal(get('treeSteps').textContent, '439 / 500');
assert.equal(get('teacherSteps').textContent, '450 / 500');
get('seek').value = '500'; get('seek').oninput();
assert.equal(get('teacherStatus').textContent, 'Completed');
get('restart').onclick();
assert.equal(get('treeSteps').textContent, '0 / 500');
assert(get('node-0').classList.contains('active'));
get('play').onclick(); frame(0); frame(20);
assert.equal(get('treeSteps').textContent, '1 / 500');
get('play').onclick(); frame(40);
assert.equal(get('treeSteps').textContent, '1 / 500');
get('examples').value = '0'; get('examples').onchange();
get('seek').value = '500'; get('seek').oninput();
assert.equal(get('treeStatus').textContent, 'Completed');
console.log('PASS: playback, pause, single-step, seek, restart, example selection, terminal freeze, and path clearing.');
console.log('These are logic tests, not browser layout or rendering tests.');
