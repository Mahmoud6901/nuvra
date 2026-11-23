const assert = require('node:assert/strict');
const {boot, flush} = require('./test_translation_ui.cjs');

const heard = 'افتح الملف';
const correction = 'ما تفتحش الملف';
const prefix = 'مسودة قبل التسجيل';
const english = 'Open the file.';
const bytes = Buffer.from([26, 69, 223, 163, 1]);
const id = 'recognition-returned-by-local-whisper';

async function voiceWorkspace(mode, learning = 'automatic') {
  const app = boot({
    status: {speechLearning: learning},
    storedMode: mode,
    provider: 'whisper',
    translate: {english},
    inlineHandler: async (url, body) => url === '/api/inline' && ({
      status: 'ready', outputText: english, workingText: body.text,
      requestId: body.requestId, revision: body.revision, issues: [], replacements: [],
    }),
  });
  await flush();
  const ctx = app.context;
  ctx.setTimeout = (fn, ms, ...args) => setTimeout(fn, ms, ...args).unref();
  class Recorder {
    static isTypeSupported() { return true; }
    start() { this.state = 'recording'; }
    stop() {
      this.state = 'inactive';
      this.ondataavailable({data: new Blob([bytes])});
      queueMicrotask(() => this.onstop());
    }
  }
  ctx.MediaRecorder = ctx.window.MediaRecorder = Recorder;
  ctx.navigator.mediaDevices.getUserMedia = async () => ({getTracks: () => [{stop() {}}]});
  ctx.URL = {createObjectURL: () => 'blob:recording', revokeObjectURL() {}};
  const uploads = [], corrections = [];
  let last = {id: null};
  let nextRecording = 0;
  const reply = data => ({ok: true, json: async () => data});
  const originalFetch = ctx.fetch;
  ctx.fetch = async (url, options) => {
    if (url === '/api/transcribe') {
      assert.ok(options.body instanceof Blob);
      const recordingId = ++nextRecording === 1 ? id : id + nextRecording;
      const capture = learning ? {learning: {id: recordingId}} : {};
      return reply({text: heard, rawText: heard, ...capture});
    }
    if (url === '/api/speech-learning') {
      const raw = Buffer.from(await options.body.arrayBuffer());
      const recordingId = options.headers['X-Relay-Learning-Id'];
      uploads.push({id: recordingId, audio: raw});
      last = {id: recordingId, text: heard, corrected: false, at: Date.now() / 1000};
      return reply({kept: true, staged: learning === 'automatic'});
    }
    if (url === '/api/speech-learning/last') return reply(last);
    if (url === '/api/speech-learning/correct') {
      const change = JSON.parse(options.body);
      corrections.push(change);
      if (last.id === change.id) last = {...last, text: change.text, corrected: true};
      return reply({saved: true});
    }
    return originalFetch(url, options);
  };
  const field = id => app.nodes.get(id);
  async function record(base) {
    field(mode === 'inline' ? 'english' : 'arabic').value = base;
    await field('record').click();
    await field('stopRecording').click();
    await flush();
  }
  function input(id, text) {
    field(id).value = text;
    field(id).listeners.input?.({type: 'input', target: field(id), isTrusted: true});
  }
  return {...app, field, record, input, uploads, corrections};
}

async function exercise(mode, learning) {
  const app = await voiceWorkspace(mode, learning);
  const source = mode === 'inline' ? 'inlineSource' : 'arabic';
  await app.record(prefix);
  assert.equal(app.field(source).value, prefix + ' ' + heard);
  if (mode === 'craft') await app.field('translate').click();
  assert.equal(app.field('english').value, english);
  if (learning === null) {
    await app.field('historyButton').click();
    await flush();
    assert.equal(app.field('correctLast').hidden, true);
    assert.deepEqual(app.uploads, []);
    assert.deepEqual(app.corrections, []);
    return;
  }
  assert.deepEqual(app.uploads, [{id, audio: bytes}]);
  app.input('english', 'English translation changed by the user.');
  app.input(source, prefix + ' ' + correction);
  await app.field('translate').click();
  await app.field('historyButton').click();
  await flush();
  assert.deepEqual(app.corrections, []);
  assert.equal(app.field('correctLast').hidden, false);
  app.field('correctLast').open = true;
  assert.equal(app.field('correctLastText').value, heard);
  assert.equal(app.field('correctLastSave').disabled, true);
  app.input('correctLastText', correction);
  await app.field('correctLastCancel').click();
  assert.deepEqual(app.corrections, []);
  await flush();
  assert.equal(app.field('correctLast').open, false);
  app.field('correctLast').open = true;
  const replacement = 'راجع تقرير المصروفات بكرة الصبح';
  app.input('correctLastText', replacement);
  assert.equal(app.field('correctLastSave').disabled, false);
  await app.record('تسجيل أحدث');
  await app.context.loadLastDictation();
  assert.equal(app.field('correctLastText').value, replacement);
  await app.field('correctLastSave').click();
  await flush();
  assert.deepEqual(app.corrections, [{id, text: replacement}]);
}

(async () => {
  for (const mode of ['craft', 'inline']) {
    for (const learning of [null, 'corrections', 'automatic']) await exercise(mode, learning);
  }
  console.log('Both writing modes keep appended text and English output out of recording corrections.');
})().catch(error => { console.error(error); process.exitCode = 1; });
