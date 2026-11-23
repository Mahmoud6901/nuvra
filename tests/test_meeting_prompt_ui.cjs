const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../src/arabic_relay/assets/meeting-prompt.html'), 'utf8');

function popup(language, app) {
  const nodes = new Map(), events = {}, calls = [];
  for (const [, attrs, id] of html.matchAll(/<([^>]+id="([^"]+)"[^>]*)>/g)) {
    nodes.set(id, {disabled: /\bdisabled\b/.test(attrs), textContent: ''});
  }
  const root = {classList: {toggle() {}}};
  const context = {
    document: {documentElement: root, getElementById: id => nodes.get(id),
      addEventListener: (event, fn) => { events[event] = fn; }},
    location: {hash: '#' + new URLSearchParams({app, token: 'window-token'})},
    localStorage: {getItem: key => key === 'relay-language' ? language : null},
    URLSearchParams, window: {addEventListener: (event, fn) => { events[event] = fn; }},
  };
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInNewContext(script, context);
  return {nodes, events, calls, root, window: context.window, ready() {
    context.window.pywebview = {api: {choose: async (...args) => { calls.push(args); }}};
    events.pywebviewready();
  }};
}

(async () => {
  for (const language of ['en', 'ar']) {
    for (const service of ['Zoom', 'Google Meet', 'Microsoft Teams']) {
      const p = popup(language, service), record = p.nodes.get('record'), dismiss = p.nodes.get('dismiss');
      assert.equal(p.root.dir, language === 'ar' ? 'rtl' : 'ltr');
      assert.equal(p.root.lang, language);
      assert.ok(p.nodes.get('question').textContent.includes(service));
      assert.equal(record.textContent, language === 'ar' ? 'تسجيل' : 'Record');
      assert.equal(dismiss.textContent, language === 'ar' ? 'تجاهل' : 'Dismiss');
      assert.equal(record.disabled, true);
      p.ready();
      assert.equal(record.disabled, false);
      assert.deepEqual(p.calls, []);
      await record.onclick();
      await record.onclick();
      await dismiss.onclick();
      assert.deepEqual(p.calls, [['record', 'window-token']]);
      p.window.promptFailed();
      assert.equal(record.disabled, false);
      assert.match(p.nodes.get('detail').textContent, language === 'ar' ? /جرّب/ : /Try again/);
      await dismiss.onclick();
      assert.deepEqual(p.calls.at(-1), ['dismiss', 'window-token']);
      const escape = popup(language, service);
      escape.ready();
      escape.events.keydown({key: 'Escape'});
      assert.deepEqual(escape.calls, [['dismiss', 'window-token']]);
    }
  }
  console.log('Meeting prompts require an explicit choice and support Record, Dismiss, Escape, and retry in both languages.');
})().catch(error => { console.error(error); process.exitCode = 1; });
