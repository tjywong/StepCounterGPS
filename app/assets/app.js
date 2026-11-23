const ui = new WebUI();
const $ = (s) => document.querySelector(s);
let labelsBuilt = false;
let recordingsKey = '';

ui.on_connect(() => { $('#conn').textContent = 'connected'; $('#conn').className = 'pill on'; });
ui.on_disconnect(() => { $('#conn').textContent = 'disconnected'; $('#conn').className = 'pill off'; });

$('#modes').addEventListener('click', (e) => {
  const mode = e.target.closest('button')?.dataset.mode;
  if (mode) ui.send_message('set_mode', { mode });
});

$('#labels').addEventListener('click', (e) => {
  const label = e.target.closest('button')?.dataset.label;
  if (label) ui.send_message('set_label', { label });
});

$('#reset-steps').addEventListener('click', () => ui.send_message('reset_steps', {}));

$('#recordings').addEventListener('change', (e) => {
  const input = e.target.closest('input[data-steps]');
  if (input) ui.send_message('set_steps', { name: input.dataset.steps, steps: input.value });
});

// Tap ✕ once to arm it ("Delete?"), again to delete. No confirm() popup: phone browsers
// can silently block those.
$('#recordings').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-delete]');
  if (!btn) return;
  if (!btn.classList.contains('armed')) {
    $('#recordings').querySelectorAll('.del.armed').forEach((b) => { b.classList.remove('armed'); b.textContent = '✕'; });
    btn.classList.add('armed');
    btn.textContent = 'Delete?';
    return;
  }
  // Leave any focused step box first, or the list won't redraw (see the status handler).
  document.activeElement?.blur();
  ui.send_message('delete_recording', { name: btn.dataset.delete });
  btn.closest('li').remove();
});

const pretty = (l) => (l[0].toUpperCase() + l.slice(1)).replace('_up', ' ↑').replace('_down', ' ↓');

function buildLabels(labels) {
  $('#labels').innerHTML = labels
    .map((l) => `<button data-label="${l}">${pretty(l)}</button>`)
    .join('');
  labelsBuilt = true;
}

function select(container, attr, value) {
  container.querySelectorAll('button').forEach((b) => b.classList.toggle('sel', b.dataset[attr] === value));
}

ui.on_message('status', (s) => {
  if (!labelsBuilt) buildLabels(s.labels);
  select($('#modes'), 'mode', s.mode);
  select($('#labels'), 'label', s.label);
  document.body.dataset.mode = s.mode;

  $('#live-record').hidden = s.mode !== 'record';
  $('#live-classify').hidden = s.mode !== 'classify';
  $('#live-idle').hidden = s.mode !== 'idle';
  $('#rate').textContent = s.rate_hz + (s.rate_hz < 40 && s.imu_health ? ` · ${s.imu_health}` : '');
  $('#error').hidden = !s.error;
  $('#error').textContent = s.error || '';

  if (s.mode === 'record') {
    $('#rec-seconds').textContent = Math.floor(s.seconds);
    $('#rec-label').textContent = s.label;
    $('#rec-file').textContent = s.file;
  }
  if (s.mode === 'classify') {
    $('#dir').textContent = s.direction;
    const p = s.prediction;
    $('#m1').textContent = p ? pretty(p.model1) : 'warming up…';
    $('#m2').textContent = p ? pretty(p.model2) : 'warming up…';
    $('#peak').textContent = p ? p.peak_g.toFixed(2) : '–';
    $('#steps').textContent = s.steps ?? 'no model';
    $('#cadence').textContent = s.steps == null ? '–' : s.cadence;
  }

  const totals = {};
  s.recordings.forEach((r) => {
    const l = s.labels.filter((x) => r.name.startsWith(x + '_')).sort((a, b) => b.length - a.length)[0];
    totals[l] = (totals[l] || 0) + 1;
  });
  $('#rec-summary').textContent = s.labels.map((l) => `${pretty(l)} ${totals[l] || 0}`).join(' · ');
  // Re-render only on change, and never under the cursor of someone typing a step count.
  const key = JSON.stringify([s.recordings, s.mode, s.file]);
  if (key === recordingsKey || document.activeElement?.matches('#recordings input')) return;
  recordingsKey = key;
  $('#recordings').innerHTML = s.recordings.length
    ? s.recordings
        .map((r) => {
          const live = s.mode === 'record' && r.name === s.file;
          return `<li><code>${r.name}</code><span>${live ? 'recording…' : r.seconds + ' s'}</span>` +
            (live ? '' : `<input type="number" min="0" inputmode="numeric" placeholder="steps" ` +
              `data-steps="${r.name}" value="${r.steps ?? ''}" title="Steps you counted">` +
              `<button class="del" data-delete="${r.name}" title="Delete">✕</button>`) + '</li>';
        })
        .join('')
    : '<li class="dim">None yet</li>';
});
