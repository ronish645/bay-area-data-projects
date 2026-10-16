// Offline unit test for report data binding and controls. This is not a browser rendering test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../reports/legacy/index.html'), 'utf8');
const context2d = new Proxy({}, { get(target, name) { return target[name] ?? (() => {}); } });
const elements = {};
for (const name of ['features', 'county', 'board', 'county-summary', 'tooltip', 'geo', 'fit']) {
  elements[name] = { value: name === 'features' ? 'combined' : 'all', innerHTML: '', textContent: '',
    children: [], append(o) { this.children.push(o); }, clientWidth: 460, clientHeight: 360,
    getContext() { return context2d; }, getBoundingClientRect() { return { left: 0, top: 0 }; } };
}
const context = {
  document: { querySelector(s) { return elements[s.slice(1)]; },
    getElementById(s) { return elements[s]; }, createElement() { return {}; } },
  window: { devicePixelRatio: 1, addEventListener() {} }, console,
};
vm.createContext(context);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
assert.equal(elements.county.children.length, 9);
assert.match(elements.board.innerHTML, /Inner-selected procedure/);
assert.match(elements['county-summary'].textContent, /1,580 tracts/);
elements.features.value = 'social';
elements.features.onchange();
assert.match(elements.board.innerHTML, /24.46/);
elements.county.value = 'Solano';
elements.county.onchange();
assert.match(elements['county-summary'].textContent, /Solano · 96 tracts/);
assert.equal(typeof elements.fit.onpointermove, 'function');
assert.equal(typeof elements.geo.onpointermove, 'function');
console.log('Report controls, county filtering and feature-family binding passed (offline unit test).');
