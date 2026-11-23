const assert = require('node:assert/strict');
const {boot, flush} = require('./test_translation_ui.cjs');

(async () => {
  const app = boot({status: {speechLearning: 'automatic'}, language: 'en'});
  const field = id => app.nodes.get(id);
  const ctx = app.context;
  ctx.setTimeout = (fn, ms, ...args) => setTimeout(fn, ms, ...args).unref();
  const base = ctx.fetch, requests = [];
  let state = {state: 'idle', available: true, promptsEnabled: false};
  let failPreference = false;
  let startGate, releaseStart;
  ctx.fetch = async (url, options) => {
    if (!url.startsWith('/api/meetings')) return base(url, options);
    requests.push({url, body: options?.body && JSON.parse(options.body)});
    if (url === '/api/meetings/prompts') {
      if (failPreference) return {ok: false, json: async () => ({error: 'Could not save preference'})};
      const {enabled} = JSON.parse(options.body);
      state = {...state, promptsEnabled: enabled};
      return {ok: true, json: async () => ({enabled})};
    }
    if (url === '/api/meetings/start') {
      await startGate;
      state = {state: 'recording', available: true, id: 'local-session', seconds: 65};
    }
    if (url === '/api/meetings/stop') state = {state: 'stopped', available: true, id: 'local-session'};
    return {ok: true, json: async () => state};
  };
  await flush();
  await field('historyButton').click();
  await flush();
  assert.equal(field('meetingControls').hidden, false);
  assert.equal(field('meetingToggle').disabled, false);
  assert.ok(requests.every(r => r.url === '/api/meetings'), 'opening the app or History must not record');
  const prompts = field('meetingPrompts');
  assert.equal(prompts.checked, false);
  prompts.checked = true;
  await prompts.listeners.change();
  assert.equal(prompts.checked, true);
  failPreference = true;
  prompts.checked = false;
  await prompts.listeners.change();
  assert.equal(prompts.checked, true);
  assert.equal(prompts.disabled, false);
  assert.deepEqual(requests.filter(r => r.url.endsWith('/start')), []);
  startGate = new Promise(resolve => { releaseStart = resolve; });
  const starting = field('meetingToggle').click();
  await field('meetingToggle').click();
  assert.equal(field('meetingToggle').disabled, true);
  releaseStart();
  await starting;
  assert.deepEqual(requests.filter(r => r.url.endsWith('/start')).map(r => r.body),
    [{microphone: true, speakers: true}]);
  assert.match(field('meetingStatus').textContent, /1:05/);
  assert.match(field('meetingToggle').textContent, /Stop/i);
  await field('meetingToggle').click();
  assert.equal(requests.filter(r => r.url.endsWith('/stop')).length, 1);
  assert.match(field('meetingToggle').textContent, /Start/i);
  const savedNotice = field('meetingStatus').textContent;
  state = {...state, partial: true, sources: {mic: {dropouts: 1}}};
  await ctx.refreshMeeting();
  assert.notEqual(field('meetingStatus').textContent, savedNotice, 'a partial archive needs a visible indication');
  state = {state: 'idle', available: false};
  await ctx.refreshMeeting();
  assert.equal(field('meetingControls').hidden, true);
  console.log('Meeting capture is manual and retries cannot start a second recording.');
})().catch(error => { console.error(error); process.exitCode = 1; });
