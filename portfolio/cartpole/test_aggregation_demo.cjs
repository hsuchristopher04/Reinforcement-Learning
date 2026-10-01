// Pure JavaScript unit tests; no browser or browser automation.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const weighted=process.argv.includes('--weighted'), capacity=process.argv.includes('--capacity'), folder=capacity?'demo-capacity':weighted?'demo-weighted':'demo-aggregation', candidate=capacity?'capacity':weighted?'weighted':'aggregated';
for(const variant of ['baseline',candidate]) {
 const html=fs.readFileSync(path.join(__dirname,folder,variant+'.html'),'utf8');
 const payload=html.match(/<script id="demo-data" type="application\/json">([\s\S]*?)<\/script>/)[1], data=JSON.parse(payload);
 const code=html.match(/<script>([\s\S]*?)<\/script>/)[1], items=new Map();
 function element(){const classes=new Set();return {children:[],value:'0',textContent:'',width:1000,height:420,
 classList:{add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x)},
 setAttribute(k,v){this[k]=v;if(k==='id')items.set(v,this)},append(...x){this.children.push(...x)},
 replaceChildren(...x){this.children=x},getContext(){return new Proxy({},{get:()=>()=>{}})}}}
 const get=id=>{if(!items.has(id))items.set(id,element());return items.get(id)};
 get('demo-data').textContent=payload;get('speed').value='1';
 vm.runInNewContext(code,{document:{getElementById:get,createElement:element,createElementNS:element,
 addEventListener(){},querySelectorAll:()=>[...items.values()].filter(x=>x.class==='node'||x.class==='edge')},requestAnimationFrame(){}});
 assert.equal(get('treeMean').textContent,data.aggregate.tree.mean_return.toFixed(2)+' / 500');
 data.episodes.forEach((ep,i)=>{get('examples').value=String(i);get('examples').onchange();
  for(const tick of [0,1,ep.tree.length,500]){get('seek').value=String(tick);get('seek').oninput();
   for(const who of ['teacher','tree']){const f=ep[who].frames[Math.min(tick,ep[who].length)];assert.equal(get(who+'Status').textContent,f.status);assert.equal(get(who+'Steps').textContent,f.step+' / 500')}
   const f=ep.tree.frames[Math.min(tick,ep.tree.length)];
   for(let n=0;n<data.tree_model.feature.length;n++)assert.equal(get('node-'+n).classList.contains('active'),f.path.includes(n));
  }
 });
 if(variant==='aggregated')assert(get('node-1').children[1].textContent.startsWith('Cart position'));
 console.log('PASS:',variant,'metrics, episode selection, path highlighting, terminal freeze, and feature labels');
}
const entry=fs.readFileSync(path.join(__dirname,folder,'index.html'),'utf8');
const fields={variant:{value:candidate+'.html'},viewer:{src:'baseline.html'}};
vm.runInNewContext(entry.match(/<script>([\s\S]*?)<\/script>/)[1],{document:{getElementById:id=>fields[id]}});
fields.variant.onchange();assert.equal(fields.viewer.src,candidate+'.html');
console.log('PASS: original/aggregated selector. Browser layout has not been checked.');
