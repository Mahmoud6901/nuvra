const assert = require('node:assert/strict');
const {boot, flush} = require('./test_translation_ui.cjs');

const status = {nativeDesktop: true, platform: 'win32', geminiConfigured: true, granolaInstalled: true, granolaRunning: true};

async function selectProviders() {
  const app = boot({language: 'en', status});
  await flush();
  await app.nodes.get('modelsButton').click();
  for (const [button, provider] of [['useGemini', 'gemini'], ['useGranola', 'granola']]) {
    await app.nodes.get(button).click();
    await flush();
    assert.equal(app.storage.get('relay-dictation-provider'), provider);
  }
  assert.ok(app.requests.every(({url}) => !url.endsWith('/start') && url !== '/api/transcribe'),
    'Selecting a speech provider must not start a recording');
}

async function granolaPaste(mode) {
  const heard = 'إيه علاقة JEV بالموضوع؟';
  const english = 'What does JEV have to do with this?';
  const app = boot({quick: true, storedMode: mode, provider: 'granola', language: 'en', status,
    translate: {english},
    inlineHandler: async (_path, body) => ({
      status: 'ready', requestId: body.requestId, revision: body.revision, workingText: body.text,
      outputText: english, issues: [], replacements: []
    })});
  let microphoneStarts = 0;
  app.context.navigator.mediaDevices.getUserMedia = async () => { microphoneStarts++; throw Error('Unexpected microphone access'); };
  app.context.window.pywebview.api.begin_capture = async () => ({captureId: 'granola-test', available: true});
  await flush();
  await app.context.window.relayQuickToggle();
  const field = app.nodes.get('flowCapture');
  assert.ok(field.focused && !field.hidden, 'Granola needs a focused field to paste its result');
  assert.match(app.nodes.get('captureLabel').textContent, /Granola/i);
  assert.equal(app.posted('native:paste').length, 0);
  field.value = heard;
  field.listeners.input({inputType: 'insertFromPaste'});
  await flush();
  assert.equal(app.nodes.get('english').value, english);
  assert.deepEqual(app.posted('native:paste').map(r => r.body.english), [english]);
  assert.equal(app.posted('native:paste')[0].body.source, 'granola');
  assert.equal(microphoneStarts, 0);
  assert.equal(app.posted('/api/transcribe').length, 0);
  assert.equal(app.posted('/api/speech-learning').length, 0);
}

async function granolaUnavailable(installation) {
  const app = boot({quick: true, provider: 'granola', language: 'en', status: {...status, ...installation}});
  let microphoneStarts = 0;
  app.context.navigator.mediaDevices.getUserMedia = async () => { microphoneStarts++; throw Error('Unexpected microphone access'); };
  await flush();
  await app.context.window.relayQuickToggle();
  await flush();
  assert.ok(!app.nodes.get('flowCapture').focused, 'A stopped or missing Granola cannot deliver to the field');
  assert.equal(microphoneStarts, 0);
  assert.equal(app.posted('native:paste').length, 0);
  assert.equal(app.posted('/api/transcribe').length, 0);
}

(async () => {
  await selectProviders();
  for (const mode of ['craft', 'inline']) await granolaPaste(mode);
  await granolaUnavailable({granolaInstalled: false, granolaRunning: false});
  await granolaUnavailable({granolaInstalled: true, granolaRunning: false});
  console.log('Granola paste handoff passed in both writing modes.');
})().catch(error => { console.error(error); process.exitCode = 1; });
