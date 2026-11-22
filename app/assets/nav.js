// Campus navigator page: record demonstrations, navigate with DAgger corrections,
// review/fix labels on a map + timeline, train, and see the held-out check.
const ui = new WebUI();
const $ = (s) => document.querySelector(s);
const ACTIONS = ['straight', 'left', 'right', 'stairs_up', 'stairs_down', 'arrived'];
const NAME = { straight: 'Straight', left: 'Turn left', right: 'Turn right',
  stairs_up: 'Stairs up', stairs_down: 'Stairs down', arrived: 'Arrived' };
const SHORT = { straight: 'Straight', left: 'Left', right: 'Right',
  stairs_up: 'Stairs ↑', stairs_down: 'Stairs ↓', arrived: 'Arrived' };
const GLYPH = { straight: '↑', left: '↰', right: '↱', stairs_up: '⇡', stairs_down: '⇣', arrived: '●' };
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let S = null;            // last status from the board
let lastSpoken = null;
let walksKey = '';
let openId = null;       // walk shown in the editor
let wasActive = false;

// ------------------------------------------------------------------ connection & tabs
ui.on_connect(() => { $('#conn').textContent = 'connected'; $('#conn').className = 'pill on'; });
ui.on_disconnect(() => { $('#conn').textContent = 'disconnected'; $('#conn').className = 'pill off'; });

function showTab(tab) {
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('sel', b.dataset.tab === tab));
  document.querySelectorAll('[data-pane]').forEach((p) => { p.hidden = p.dataset.pane !== tab; });
  store.set('nav.tab', tab);
  if (tab === 'label') requestAnimationFrame(drawEditor);
}
$('#tabs').addEventListener('click', (e) => {
  const t = e.target.closest('button')?.dataset.tab;
  if (t) showTab(t);
});
showTab(store.get('nav.tab') || 'walk');

// ------------------------------------------------------------------ walk tab
$('#walker').value = store.get('nav.walker') || '';
$('#walker').addEventListener('change', () => store.set('nav.walker', $('#walker').value.trim()));

// ------------------------------------------------------------------ routes
// The board keeps the route list and the current route, so every phone sees the same one.
const curRoute = () => S?.routes.find((r) => r.name === S.route) || null;
let routesKey = '';

function routeLabel(r) {   // short: phones cut long <option>s. ✓ trained, ↻ needs retraining
  return `${!r.trained ? '' : r.stale ? '↻ ' : '✓ '}${r.name} (${r.walks})`;
}
function renderRoutes(s) {
  const key = JSON.stringify([s.routes.map(routeLabel), s.route]);
  if (key !== routesKey && document.activeElement !== $('#route-select')) {
    routesKey = key;
    $('#route-select').innerHTML = s.routes.length
      ? s.routes.map((r) => `<option value="${esc(r.name)}" ${r.name === s.route ? 'selected' : ''}>${esc(routeLabel(r))}</option>`).join('')
      : '<option value="">No routes yet</option>';
    if (openId && ED.walk && ED.walk.route !== s.route) closeEditor();
  }
  const r = curRoute();
  $('#route-state').innerHTML = !r ? 'Create a route to start: ＋ New.'
    : `${r.ticked}/${s.max_walks} walks ticked for training · ` +
      (r.length_m ? `${r.length_m} m` : 'length unknown') + ` · turns ≥ ${r.turn_min}° · ` +
      (!r.trained ? '<span class="warn">not trained</span>'
        : r.stale ? `<span class="warn">walks or threshold changed since training ${esc(r.trained)}: train again</span>`
          : `trained ${esc(r.trained)}`);
}
$('#route-select').addEventListener('change', (e) => {
  const name = e.target.value;
  if (S?.nav && S.nav.route !== name &&
      !confirm(`A walk on '${S.nav.route}' is in progress. Stop and save it, and switch to '${name}'?`)) {
    routesKey = '';               // put the select back
    renderRoutes(S);
    return;
  }
  ui.send_message('route_select', { name });
  e.target.blur();
});
$('#route-new-btn').addEventListener('click', () => {
  $('#route-new').hidden = !$('#route-new').hidden;
  if (!$('#route-new').hidden) $('#new-name').focus();
});
$('#new-cancel').addEventListener('click', () => { $('#route-new').hidden = true; });
$('#route-new').addEventListener('submit', (e) => {
  e.preventDefault();
  const name = $('#new-name').value.trim();
  if (!name) { flash('Give the route a name.'); return; }
  if (S?.routes.some((r) => r.name === name)) { flash(`There is already a route called '${name}'.`); return; }
  if (S?.nav && !confirm(`A walk on '${S.nav.route}' is in progress. Stop and save it, and switch to the new route?`)) return;
  ui.send_message('route_create', { name, length_m: parseFloat($('#new-len').value) || null });
  $('#new-name').value = ''; $('#new-len').value = '';
  $('#route-new').hidden = true;
});

function start(kind) {
  if (!S?.route) { flash('Create a route first (＋ New).'); return; }
  speak(null);   // unlock speech on iOS: it must start from a tap
  ui.send_message('nav_start', { kind, route: S.route, walker: $('#walker').value.trim() || 'me' });
}
$('#start-demo').addEventListener('click', () => start('demo'));
$('#start-nav').addEventListener('click', () => start('dagger'));
$('#stop').addEventListener('click', () => ui.send_message('nav_stop', {}));
$('#marks').addEventListener('click', (e) => {
  const kind = e.target.closest('button')?.dataset.mark;
  if (kind) { ui.send_message('nav_mark', { kind }); buzz(60); }
});
$('#correct').addEventListener('click', (e) => {
  const action = e.target.closest('button')?.dataset.action;
  if (action) { ui.send_message('nav_correct', { action }); buzz(60); }
});
$('#auto').addEventListener('click', () => ui.send_message('nav_correct', { action: null }));
$('#speak').checked = store.get('nav.speak') !== '0';
$('#speak').addEventListener('change', () => store.set('nav.speak', $('#speak').checked ? '1' : '0'));

function buzz(ms) { try { navigator.vibrate?.(ms); } catch { /* unsupported */ } }
function speak(text) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text ?? '');
  if (text == null) u.volume = 0;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

let msgTimer = null;
function flash(text) {
  $('#msg').textContent = text;
  $('#msg').hidden = !text;
  clearTimeout(msgTimer);
  if (text) msgTimer = setTimeout(() => { $('#msg').hidden = true; }, 6000);
}
let lastMsg = null;

function renderWalk(s) {
  const n = s.nav;
  const active = !!n;
  $('#setup').hidden = active;
  $('#live').hidden = !active;
  const r = curRoute();
  $('#walk-count').textContent = r ? `Training walks on this route: ${r.ticked} / ${s.max_walks} (demos + DAgger runs, both walkers).` : '';
  $('#no-route').hidden = !!r;
  $('#start-demo').disabled = !r;
  $('#start-nav').disabled = !r || !r.trained;
  const me = s.calibration?.[$('#walker').value.trim() || 'me'];
  $('#cal-line').innerHTML = me && me.source !== 'default'
    ? `Your stride is calibrated (${esc(me.source)}, ${me.walks} walk${me.walks === 1 ? '' : 's'}).`
    : '<span class="warn">Not calibrated yet for this walker: distances use a 0.70 m default step.</span>';
  if (!active) { lastSpoken = null; wasActive = false; return; }
  if (!wasActive) showTab('walk');   // a session started (maybe from another phone): show it
  wasActive = true;

  $('#live-kind').textContent = n.kind === 'demo' ? '● Demo' : '▶ Navigating';
  $('#live-kind').className = 'pill ' + (n.kind === 'demo' ? 'rec' : 'on');
  $('#live-route').textContent = `${n.route} · ${n.walker}`;
  $('#calib').hidden = !n.calibrating;
  $('#instr').hidden = n.calibrating || n.kind === 'demo';
  $('#correct-box').hidden = n.kind !== 'dagger';
  $('#s-dist').textContent = (n.calibrated ? '' : '≈') + n.dist.toFixed(0);
  $('#s-dist-unit').textContent = `m · ${n.step_m} m/step`;
  const stairs = n.activity === 'stairs_up' ? '⇡ on stairs (up)' : n.activity === 'stairs_down' ? '⇣ on stairs (down)' : '';
  $('#s-activity').textContent = stairs || (n.activity ? `classifier: ${n.activity}` : '');
  $('#s-steps').textContent = n.steps;
  $('#s-head').textContent = n.heading;
  $('#s-turns').textContent = n.turns;
  const m = n.marks[n.marks.length - 1];
  $('#last-mark').textContent = m ? `last tap: ${m.kind.replace('_', ' ')} at ${m.t.toFixed(0)} s` : '';

  if (n.kind === 'dagger' && !n.calibrating) {
    const a = n.action;
    $('#instr').dataset.action = a;
    $('#instr-glyph').textContent = GLYPH[a];
    $('#instr-text').textContent = NAME[a];
    $('#instr-who').textContent = n.override
      ? `your correction (policy says ${SHORT[n.policy_action] ?? '–'})`
      : `policy · ${Math.round(n.confidence * 100)}% of neighbours agree`;
    document.querySelectorAll('#correct button').forEach((b) => b.classList.toggle('sel', b.dataset.action === n.override));
    $('#auto').hidden = !n.override;
    if (a !== lastSpoken) {
      if (lastSpoken !== null && $('#speak').checked) { speak(NAME[a]); buzz(a === 'straight' ? 80 : [150, 80, 150]); }
      lastSpoken = a;
    }
  }
}

// ------------------------------------------------------------------ label tab: walk list
$('#walks').addEventListener('click', (e) => {
  const li = e.target.closest('li[data-id]');
  if (!li) return;
  const id = li.dataset.id;
  if (e.target.closest('.del')) {
    if (confirm('Delete this walk? Its IMU data goes too.')) ui.send_message('walk_delete', { id });
    if (openId === id) closeEditor();
    return;
  }
  if (e.target.matches('input[type=checkbox]')) {
    ui.send_message('walk_include', { id, include: e.target.checked });
    return;
  }
  ui.send_message('walk_get', { id });
});

function renderWalks(s) {
  const walks = s.walks.filter((w) => w.route === s.route);
  const key = JSON.stringify([walks, openId, s.route]);
  if (key === walksKey) return;
  walksKey = key;
  const used = walks.filter((w) => w.include).length;
  $('#walks-summary').textContent = `${used}/${s.max_walks} ticked for training`;
  $('#walks').innerHTML = walks.length ? walks.map((w) => `
    <li data-id="${esc(w.id)}" class="${w.id === openId ? 'open' : ''}">
      <input type="checkbox" ${w.include ? 'checked' : ''} title="Use for training">
      <div class="w-main">
        <div><b>${esc(w.walker)}</b> · ${esc(w.created || w.id.split('_')[0])}</div>
        <div class="sub dim">${w.kind === 'dagger' ? 'DAgger run' : 'demo'} · ${w.seconds} s · ${w.steps} steps · ${w.distance_m ? `${w.distance_m} m walked, est. ${w.dist} m` : `≈${w.dist} m`}
          ${w.corrections ? ` · ${w.corrections} corrections` : ''}${w.reviewed ? '' : ' · <span class="warn">labels not reviewed</span>'}</div>
      </div>
      <button class="del" title="Delete">✕</button>
    </li>`).join('') : `<li class="dim">No walks of ${s.route ? `'${esc(s.route)}'` : 'this route'} yet. Record a demo on the Walk tab.</li>`;
}

// ------------------------------------------------------------------ label tab: editor
const ED = { walk: null, turns: [], t: [], lab: [], pol: [], exp: [], xy: [], zoom: 1, sel: null };

function closeEditor() { openId = null; ED.walk = null; $('#editor').hidden = true; walksKey = ''; }

ui.on_message('walk', (m) => {
  const w = m.walk;
  const cols = w.trace.cols;
  const ix = Object.fromEntries(cols.map((c, i) => [c, i]));
  const rows = w.trace.rows.filter((r) => !r[ix.calibrating]);
  if (!rows.length) { flash('That walk has no data after calibration.'); return; }
  openId = w.id;
  ED.walk = w;
  ED.turns = m.turns;
  ED.t = rows.map((r) => r[ix.t]);
  ED.k = m.k;
  ED.swing = rows.map((r) => r[ix.swing]);
  ED.act = rows.map((r) => (r[ix.activity] === 'stairs_up' || r[ix.activity] === 'stairs_down') ? r[ix.activity] : null);
  ED.head = rows.map((r) => r[ix.heading]);
  ED.pol = rows.map((r) => ACTIONS[r[ix.policy]] ?? null);
  ED.exp = rows.map((r) => ACTIONS[r[ix.expert]] ?? null);
  ED.lab = rowsFromSegments(w.labels);
  ED.sel = null;
  // Dead-reckoned path: start pointing up the screen, left turns curve left.
  let x = 0, y = 0;
  ED.xy = rows.map((r, i) => {
    if (i) {
      const d = (r[ix.swing] - rows[i - 1][ix.swing]) * m.k;
      const h = r[ix.heading] * Math.PI / 180;
      x -= d * Math.sin(h);
      y -= d * Math.cos(h);
    }
    return [x, y];
  });
  $('#ed-title').textContent = `${w.route} · ${w.walker}`;
  $('#ed-info').textContent = `${w.kind === 'dagger' ? 'DAgger run' : 'Demo'} ${w.created} · ` +
    `${ED.t[ED.t.length - 1].toFixed(0)} s · turns ≥ ${m.turn_min}°` +
    (m.proposed ? ' · new auto-labels, not saved yet' : (w.reviewed ? ' · reviewed' : ' · auto-labels, not reviewed'));
  $('#ed-dist').value = w.distance_m ?? '';
  const est = ED.swing[ED.swing.length - 1] * m.k;
  $('#ed-est').textContent = `estimate: ${est.toFixed(0)} m` +
    (m.cal && m.cal.source !== 'default' ? ` (stride ${m.cal.source})` : ' (uncalibrated default)');
  $('#editor').hidden = false;
  walksKey = '';
  if (S) renderWalks(S);
  showTab('label');
  drawEditor();
  if (S?.last_saved === w.id && !m.proposed && !w.reviewed) {
    flash('Walk saved. Check its labels below, then press Save & use for training.');
  }
});

function rowsFromSegments(segs) {
  const lab = ED.t.map(() => 'straight');
  for (const s of segs) ED.t.forEach((t, i) => { if (t >= s.t0 && t <= s.t1) lab[i] = s.action; });
  return lab;
}
function segmentsFromRows(lab) {
  const out = [];
  for (let i = 0; i < lab.length;) {
    let j = i;
    while (j + 1 < lab.length && lab[j + 1] === lab[i]) j++;
    if (lab[i] !== 'straight') out.push({ t0: ED.t[i], t1: ED.t[j], action: lab[i] });
    i = j + 1;
  }
  return out;
}

const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();
const color = (a) => css(`--a-${a}`);

function fitCanvas(c, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  c.width = w * dpr; c.height = h * dpr; c.style.height = h + 'px';
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

function drawEditor() {
  if (!ED.walk || $('#editor').hidden) return;
  drawMap();
  drawTimeline();
  const sel = ED.sel;
  if (sel) {
    const [a, b] = [Math.min(sel[0], sel[1]), Math.max(sel[0], sel[1])];
    const d = (ED.swing[b] - ED.swing[a]) * ED.k;
    $('#sel-info').textContent = `Selected ${ED.t[a].toFixed(1)}–${ED.t[b].toFixed(1)} s (${d.toFixed(1)} m), ` +
      `currently ${[...new Set(ED.lab.slice(a, b + 1))].map((x) => SHORT[x]).join(' + ')}`;
  } else {
    $('#sel-info').textContent = 'Nothing selected.';
  }
  $('#paint').innerHTML = ACTIONS.map((a) =>
    `<button data-paint="${a}" ${sel ? '' : 'disabled'}><i style="background:${color(a)}"></i>${SHORT[a]}</button>`).join('');
}

function drawMap() {
  const [g, w, h] = fitCanvas($('#map'), Math.min(320, $('#map').clientWidth * 0.75));
  g.clearRect(0, 0, w, h);
  const xs = ED.xy.map((p) => p[0]), ys = ED.xy.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pad = 14;
  const sc = Math.min((w - 2 * pad) / Math.max(1, x1 - x0), (h - 2 * pad) / Math.max(1, y1 - y0));
  const ox = (w - (x1 - x0) * sc) / 2, oy = (h - (y1 - y0) * sc) / 2;
  ED.toScreen = (p) => [ox + (p[0] - x0) * sc, oy + (p[1] - y0) * sc];
  const sel = ED.sel && [Math.min(...ED.sel), Math.max(...ED.sel)];
  g.lineCap = 'round';
  for (let i = 1; i < ED.xy.length; i++) {
    const [ax, ay] = ED.toScreen(ED.xy[i - 1]), [bx, by] = ED.toScreen(ED.xy[i]);
    const inSel = sel && i >= sel[0] && i <= sel[1];
    g.strokeStyle = color(ED.lab[i]);
    g.lineWidth = inSel ? 9 : (ED.lab[i] === 'straight' ? 3 : 6);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
  }
  const dot = (p, c) => { const [x, y] = ED.toScreen(p); g.fillStyle = c; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); };
  dot(ED.xy[0], css('--ok'));
  dot(ED.xy[ED.xy.length - 1], css('--rec'));
  g.fillStyle = css('--dim'); g.font = '11px system-ui';
  g.fillText(`${Math.round(Math.max(x1 - x0, y1 - y0))} m across`, 6, h - 6);
}

const TL = { h: 150, top: 8, headH: 70, bandY: 92, bandH: 22, polY: 118, subH: 10, expY: 132 };

function tlGeometry() {
  const c = $('#timeline');
  const vw = c.clientWidth;
  const full = vw * ED.zoom;
  const off = (full - vw) * ($('#pan').value / 1000);
  const T = ED.t[ED.t.length - 1] || 1, T0 = ED.t[0];
  return { vw, full, off, tx: (t) => ((t - T0) / (T - T0)) * full - off, xt: (x) => T0 + ((x + off) / full) * (T - T0) };
}

function drawTimeline() {
  const [g, w, h] = fitCanvas($('#timeline'), ED.walk.kind === 'dagger' ? TL.h : TL.polY + 4);
  const G = tlGeometry();
  $('#pan').hidden = ED.zoom <= 1;
  g.clearRect(0, 0, w, h);
  const tag = (text, y) => {
    g.font = '10px system-ui';
    g.globalAlpha = 0.6;   // see-through: never hide the start of a band
    g.fillStyle = css('--bg'); g.fillRect(0, y - 9, g.measureText(text).width + 8, 12);
    g.globalAlpha = 1;
    g.fillStyle = css('--text'); g.fillText(text, 4, y);
  };
  // time grid
  g.strokeStyle = css('--line'); g.fillStyle = css('--dim'); g.font = '10px system-ui'; g.lineWidth = 1;
  const span = (ED.t[ED.t.length - 1] - ED.t[0]) / ED.zoom;
  const step = [5, 10, 20, 30, 60, 120].find((s) => span / s < 8) || 300;
  for (let t = Math.ceil(ED.t[0] / step) * step; t <= ED.t[ED.t.length - 1]; t += step) {
    const x = G.tx(t);
    if (x < 0 || x > w) continue;
    g.beginPath(); g.moveTo(x, TL.top); g.lineTo(x, TL.top + TL.headH); g.stroke();
    g.fillText(`${t}s`, x + 2, TL.top + 9);
  }
  // heading
  const hs = ED.head, hmin = Math.min(...hs), hmax = Math.max(...hs);
  const hy = (v) => TL.top + TL.headH - ((v - hmin) / Math.max(30, hmax - hmin)) * (TL.headH - 6);
  g.strokeStyle = css('--text'); g.lineWidth = 1.5; g.beginPath();
  ED.t.forEach((t, i) => { const x = G.tx(t); i ? g.lineTo(x, hy(hs[i])) : g.moveTo(x, hy(hs[i])); });
  g.stroke();
  tag('heading (left = up)', TL.top + TL.headH - 2);
  // detected turns
  g.fillStyle = css('--text');
  for (const tr of ED.turns) {
    const x = G.tx((tr.t_start + tr.t_end) / 2);
    g.beginPath(); g.moveTo(x, TL.bandY - 2); g.lineTo(x - 5, TL.bandY - 10); g.lineTo(x + 5, TL.bandY - 10); g.fill();
  }
  // bands
  const band = (vals, y, hh) => {
    for (let i = 0; i < vals.length; i++) {
      if (!vals[i]) continue;
      const x0 = G.tx(i ? (ED.t[i - 1] + ED.t[i]) / 2 : ED.t[0]);
      const x1 = G.tx(i < vals.length - 1 ? (ED.t[i] + ED.t[i + 1]) / 2 : ED.t[i]);
      g.fillStyle = color(vals[i]);
      g.fillRect(x0, y, Math.max(1, x1 - x0 + 0.5), hh);
    }
  };
  band(ED.lab, TL.bandY, TL.bandH);
  if (ED.act.some(Boolean)) {
    band(ED.act, TL.bandY + TL.bandH + 1, 4);   // thin strip under the labels: classifier's stairs
  }
  if (ED.walk.kind === 'dagger') {
    band(ED.pol, TL.polY, TL.subH);
    band(ED.exp, TL.expY, TL.subH);
    tag('policy said', TL.polY + 9);
    tag('you corrected', TL.expY + 9);
  }
  tag('label', TL.bandY + 15);
  // selection
  if (ED.sel) {
    const a = Math.min(...ED.sel), b = Math.max(...ED.sel);
    const x0 = G.tx(ED.t[a]) - 1, x1 = G.tx(ED.t[b]) + 1;
    g.fillStyle = css('--sel'); g.fillRect(x0, 0, x1 - x0, h);
    g.strokeStyle = css('--accent'); g.lineWidth = 2; g.strokeRect(x0, 1, x1 - x0, h - 2);
  }
}

function rowAt(t) {
  let lo = 0, hi = ED.t.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ED.t[m] < t) lo = m + 1; else hi = m; }
  return lo;
}
function selectRun(i) {   // the contiguous same-label stretch around row i
  let a = i, b = i;
  while (a > 0 && ED.lab[a - 1] === ED.lab[i]) a--;
  while (b < ED.lab.length - 1 && ED.lab[b + 1] === ED.lab[i]) b++;
  ED.sel = [a, b];
}

let drag = null;
$('#timeline').addEventListener('pointerdown', (e) => {
  if (!ED.walk) return;
  e.target.setPointerCapture(e.pointerId);
  const i = rowAt(tlGeometry().xt(e.offsetX));
  drag = { x: e.offsetX, i };
});
$('#timeline').addEventListener('pointermove', (e) => {
  if (!drag) return;
  if (Math.abs(e.offsetX - drag.x) < 5 && !drag.moved) return;
  drag.moved = true;
  ED.sel = [drag.i, rowAt(tlGeometry().xt(e.offsetX))];
  drawEditor();
});
$('#timeline').addEventListener('pointerup', (e) => {
  if (drag && !drag.moved) selectRun(drag.i);
  drag = null;
  drawEditor();
});
$('#map').addEventListener('click', (e) => {
  if (!ED.walk) return;
  let best = 0, bd = Infinity;
  ED.xy.forEach((p, i) => {
    const [x, y] = ED.toScreen(p);
    const d = (x - e.offsetX) ** 2 + (y - e.offsetY) ** 2;
    if (d < bd) { bd = d; best = i; }
  });
  selectRun(best);
  drawEditor();
});
$('#paint').addEventListener('click', (e) => {
  const a = e.target.closest('button')?.dataset.paint;
  if (!a || !ED.sel) return;
  const [i, j] = [Math.min(...ED.sel), Math.max(...ED.sel)];
  for (let k = i; k <= j; k++) ED.lab[k] = a;
  selectRun(i);
  drawEditor();
});
$('#zoom-in').addEventListener('click', () => { ED.zoom = Math.min(16, ED.zoom * 2); drawEditor(); });
$('#zoom-out').addEventListener('click', () => { ED.zoom = Math.max(1, ED.zoom / 2); drawEditor(); });
$('#pan').addEventListener('input', drawEditor);
$('#autolabel').addEventListener('click', () => {
  if (ED.walk) ui.send_message('walk_autolabel', { id: ED.walk.id, lead_m: parseFloat($('#lead').value) || 0,
    distance_m: parseFloat($('#ed-dist').value) || null });
});
$('#save').addEventListener('click', () => {
  if (!ED.walk) return;
  ui.send_message('walk_save', { id: ED.walk.id, labels: segmentsFromRows(ED.lab), include: true,
    distance_m: parseFloat($('#ed-dist').value) || null });
  flash('Saved.');
});
window.addEventListener('resize', () => requestAnimationFrame(drawEditor));

// ------------------------------------------------------------------ train tab
$('#train').addEventListener('click', () => { if (S?.route) ui.send_message('train', { route: S.route }); });
$('#route-save').addEventListener('click', () => {
  const r = curRoute();
  if (!r) return;
  ui.send_message('route_update', { name: r.name, new_name: $('#set-name').value.trim(),
    length_m: parseFloat($('#set-len').value) || null, turn_min: parseFloat($('#turn-min').value) || r.turn_min });
  settingsFor = null;              // refill from the board's answer
});
$('#route-delete').addEventListener('click', () => {
  const r = curRoute();
  if (r && confirm(`Delete '${r.name}', its ${r.walks} walks and its training? This can't be undone.`)) {
    ui.send_message('route_delete', { name: r.name });
  }
});
let settingsFor = null;            // route whose settings are in the form (don't overwrite while typing)
const pct = (v) => (v > 0 ? '+' : '') + Math.round(v * 100) + '%';

function renderTrain(s) {
  const r = curRoute();
  document.querySelectorAll('[data-pane=train] .card').forEach((c) => { c.hidden = !r; });
  $('#train').textContent = r ? `Train '${r.name}'` : 'Train this route';
  $('#train').disabled = !r || !r.ticked;
  $('#policy-info').innerHTML = !r ? ''
    : r.trained
      ? `Trained ${esc(r.trained)}: ${r.examples} examples, turns ≥ ${r.policy_turn_min}°.` +
        (r.stale ? ' <span class="warn">Walks or threshold changed since: train again.</span>' : '')
      : `Not trained yet. Tick reviewed walks on the Label tab (${r.ticked} so far), then train.`;
  // Refill the settings form when the route changes or the board's values change, unless editing.
  const editing = ['#set-name', '#set-len', '#turn-min'].some((id) => document.activeElement === $(id));
  const fill = r ? JSON.stringify([r.name, r.length_m, r.turn_min]) : null;
  if (!editing && fill !== settingsFor) {
    settingsFor = fill;
    $('#set-name').value = r?.name ?? '';
    $('#set-len').value = r?.length_m ?? '';
    $('#turn-min').value = r?.turn_min ?? 45;
  }
  $('#eval-state').textContent = s.evaluating === s.route ? 'running…' : '';
  const cal = Object.entries(s.calibration || {});
  $('#cal').innerHTML = cal.length
    ? '<tr><th>walker</th><th>from</th><th>step at 60° swing</th></tr>' + cal.map(([w, c]) =>
      `<tr><td>${esc(w)}</td><td>${esc(c.source)}${c.walks ? ` · ${c.walks}` : ''}</td><td>${(c.k * 60).toFixed(2)} m</td></tr>`).join('')
    : '<tr><td class="dim">No walks yet.</td></tr>';
  const ev = s.eval?.[s.route];
  const key = JSON.stringify([s.route, ev]);
  if (key === $('#eval').dataset.key) return;
  $('#eval').dataset.key = key;
  if (!ev) {
    $('#eval').innerHTML = r?.trained ? '<tr><td class="dim">Press train to run the check.</td></tr>' : '';
    return;
  }
  const distNote = (r) => r.dist_err_swing === undefined ? ''
    : `<div class="sub dim">distance error: swing ${pct(r.dist_err_swing)}, fixed step ${pct(r.dist_err_steps)}</div>`;
  $('#eval').innerHTML = '<tr><th>walk</th><th>match</th><th>maneuvers</th><th>false</th></tr>' + ev.map((r) => r.skipped
    ? `<tr><td>${esc(r.walker ?? '')}<div class="sub dim">${esc(r.id)}</div>${distNote(r)}</td><td colspan="3" class="dim">${esc(r.skipped)}</td></tr>`
    : `<tr><td>${esc(r.walker)}<div class="sub dim">${esc(r.id)}</div>${distNote(r)}</td>` +
      `<td>${Math.round(r.accuracy * 100)}%</td><td>${r.hit}/${r.maneuvers}</td><td>${r.false_alarms}</td></tr>`).join('');
}

// ------------------------------------------------------------------ status
ui.on_message('status', (s) => {
  S = s;
  if (s.nav_msg && s.nav_msg !== lastMsg) flash(s.nav_msg);
  lastMsg = s.nav_msg;
  renderRoutes(s);
  renderWalk(s);
  renderWalks(s);
  renderTrain(s);
});
