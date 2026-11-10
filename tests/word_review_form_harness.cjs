// Minimal DOM harness for the form's state/export logic; not a browser render test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[2], 'utf8');
const data = source.match(/<script id="review-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = source.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const draft = new Map();
function run() {
  let downloaded = null;
  const cards = Array.from({length:32}, () => {
    const fields = {'.transcription':{value:'',disabled:false}, '.unreadable':{checked:false},
      '.visual':{value:''}};
    return {fields, done:false, listeners:{}, querySelector(selector){return fields[selector]},
      classList:{toggle(){}}, addEventListener(type, callback){this.listeners[type]=callback}};
  });
  const elements = {'review-data':{textContent:data}, download:{disabled:true,listeners:{},
    addEventListener(type, callback){this.listeners[type]=callback}}, status:{textContent:''},
    count:{textContent:''}, progress:{value:0}};
  const context = vm.createContext({
    document:{getElementById(id){return elements[id]}, querySelectorAll(){return cards},
      body:{appendChild(){}}, createElement(){return {click(){},remove(){}}}},
    localStorage:{getItem(key){return draft.get(key)},setItem(key,value){draft.set(key,value)}},
    Blob, URL:{createObjectURL(blob){downloaded=blob;return 'blob:test'},revokeObjectURL(){}},
    setTimeout(callback){callback()},
  });
  vm.runInContext(script,context);
  return {cards,elements,getDownload(){return downloaded}};
}
(async () => {
  const first = run();
  assert.equal(first.elements.download.disabled,true);
  first.cards.slice(0,31).forEach(card=>{
    card.fields['.unreadable'].checked=true;
    card.fields['.visual'].value='no';
    card.listeners.change();
  });
  assert.equal(first.elements.progress.value,31);
  first.elements.download.listeners.click();
  assert.equal(first.getDownload(),null);
  const restored=run();
  assert.equal(restored.elements.progress.value,31);
  const last=restored.cards[31];
  last.fields['.unreadable'].checked=true;
  last.fields['.visual'].value='no';
  last.listeners.change();
  assert.equal(last.fields['.transcription'].disabled,true);
  assert.equal(restored.elements.download.disabled,false);
  restored.elements.download.listeners.click();
  const output=await restored.getDownload().text();
  const payload=JSON.parse(output);
  assert.equal(payload.answers.length,32);
  assert(payload.answers.every(row=>row.transcription==='<unreadable>' && row.no_visual_defects==='no'));
  assert.equal(payload.checkpoint_sha256,JSON.parse(data).checkpoint_sha256);
  fs.writeFileSync(process.argv[3],output,'utf8');
  process.stdout.write('form checks passed\n');
})().catch(error=>{process.stderr.write(String(error));process.exitCode=1});
