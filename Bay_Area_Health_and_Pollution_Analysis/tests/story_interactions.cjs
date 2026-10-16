// Offline data-binding tests. These do not claim a real browser visual rendering test.
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path'), assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '../reports/index.html'), 'utf8');
const ctx = new Proxy({}, {get(t,k) {return t[k] ?? (()=>{});}});
const el = {};
for (const id of ['metric','rank-explanation','rankings','features','model-table','county','county-stat','geo','fit','point-detail']) {
  el[id] = {value:id==='metric'?'PM2.5':id==='features'?'combined':'all',innerHTML:'',textContent:'',children:[],
    append(o) {this.children.push(o);},clientWidth:460,clientHeight:330,getContext() {return ctx;},getBoundingClientRect() {return {left:0,top:0};}};
}
const sandbox={document:{getElementById:id=>el[id],createElement:()=>({})},window:{devicePixelRatio:1,addEventListener() {}},console};
vm.createContext(sandbox);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],sandbox);
assert.equal(el.county.children.length,9);
assert.ok(el.rankings.innerHTML.indexOf('Napa') < el.rankings.innerHTML.indexOf('Alameda'));
el.metric.value='Asthma';el.metric.onchange();
assert.ok(el.rankings.innerHTML.indexOf('Solano') < el.rankings.innerHTML.indexOf('Alameda'));
el.metric.value='Pollution Burden';el.metric.onchange();
assert.ok(el.rankings.innerHTML.indexOf('Alameda') < el.rankings.innerHTML.indexOf('Napa'));
assert.match(el['county-stat'].textContent,/1,580 labeled tracts/);
el.features.value='social';el.features.onchange();assert.match(el['model-table'].innerHTML,/25.08/);
el.county.value='Solano';el.county.onchange();assert.match(el['county-stat'].textContent,/Solano · 96 labeled tracts/);
assert.equal(typeof el.fit.onpointermove,'function');assert.equal(typeof el.geo.onclick,'function');
assert.match(html,/exploratory, not an untouched confirmatory test/);
console.log('Bay Area rankings, feature families, county filtering, interaction handlers and evaluation status passed.');
