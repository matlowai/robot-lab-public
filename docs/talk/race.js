/* Race player: one video of three stacked patrol lanes, with a lane tracker beside each row,
   checkpoint ticks, a finish dot per lane and a scrubbable master timeline.
   Usage: <div class="race" data-src="media/race_seed21.mp4" data-poster="media/race_seed21_still.jpg" data-rate="8"></div>
   then RacePlayer.mountAll(). Colours come from the host page's tokens (--paper, --muted, --line, --panel,
   --hazard, --sensor, --alarm) with fallbacks. Data: patrol-lab runs M1-SCN-AV-021-{control,heuristic,rl}. */
(function () {
  var RACE = {
    duration: 352.6,
    lanes: [
      { name: 'No avoidance', color: 'var(--muted,#9AA8BA)',
        cps: [50.4, 72.4, 112.3, 144.3, 185.1, 219.7, 268.4],
        fell: 295.0, fellNote: 'knocked over' },
      { name: 'Hand-written planner', color: 'var(--sensor,#5CC8D6)',
        cps: [46.6, 66.5, 105.1, 135.1, 172.7, 204.2, 249.6, 283.1],
        done: 302.3, fell: 314.0, fellNote: 'knocked over while parked' },
      { name: 'Learned policy (RL)', color: 'var(--hazard,#F2B33D)',
        cps: [44.4, 63.9, 100.1, 128.6, 165.3, 195.6, 239.5, 273.2],
        done: 292.4 }
    ],
    jumpTo: 270,
    // Shown over the lane trackers once the last event has played (DECISIONS.md:67 in patrol-lab).
    lesson: {
      at: 315.5,
      title: 'What this race really tested',
      points: [
        'The walkers followed a fixed timetable, fitted to the slow robot with no avoidance.',
        'The planner and the learned policy ran ahead, so they passed every encounter before its walker arrived. They never faced the test.',
        'The only contacts were late walkers reaching the charger after the robots had parked. One knocked the parked planner over.',
        'Fix: pedestrians that react to whichever robot is in front of them. Every planner-vs-RL number before that was confounded.'
      ]
    }
  };
  var NCP = 8;

  function fmt(t) { t = Math.max(0, t); var m = Math.floor(t / 60), s = t - m * 60; return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1); }
  function fmt0(t) { t = Math.max(0, t); var m = Math.floor(t / 60), s = Math.floor(t - m * 60); return m + ':' + (s < 10 ? '0' : '') + s; }
  function pct(t) { return (100 * t / RACE.duration).toFixed(3) + '%'; }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  var CSS = '' +
    '.rp{--rp-fg:var(--paper,#EEF1F4);--rp-mute:var(--muted,#9AA8BA);--rp-line:var(--line,#26364D);--rp-panel:var(--panel,#152234);--rp-alarm:var(--alarm,#E8664A);' +
    'color:var(--rp-fg);font-family:inherit;display:flex;flex-direction:column;gap:18px;width:100%}' +
    '.rp-top{display:flex;gap:28px;align-items:stretch;min-width:0}' +
    '.rp-vid{position:relative;flex:none;height:var(--rp-h,640px);aspect-ratio:1280/1080;max-width:100%}' +
    '.rp-vid video{display:block;width:100%;height:100%;object-fit:contain;background:#000;border-radius:10px;border:1px solid var(--rp-line);cursor:pointer}' +
    '.rp-lanes{position:relative;flex:1;min-width:0;overflow:hidden;display:grid;grid-template-rows:repeat(3,1fr);gap:0}' +
    '.rp-lesson{position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;justify-content:flex-start;gap:calc(var(--rp-fs,26px)*.45);padding:calc(var(--rp-fs,26px)*.9);' +
    'background:var(--rp-panel);border:1px solid var(--hazard,#F2B33D);border-radius:12px;opacity:0;pointer-events:none;transform:translateY(8px);transition:opacity .35s,transform .35s}' +
    '.rp-lesson.on{opacity:1;pointer-events:auto;transform:none}' +
    '.rp-lesson .k{font-family:var(--mono,"JetBrains Mono",monospace);font-size:calc(var(--rp-fs,26px)*.62);letter-spacing:.14em;text-transform:uppercase;color:var(--hazard,#F2B33D)}' +
    '.rp-lesson h4{margin:0;font-size:calc(var(--rp-fs,26px)*1.15);line-height:1.15}' +
    '.rp .rp-lesson ol{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:calc(var(--rp-fs,26px)*.45);font-size:calc(var(--rp-fs,26px)*.98);line-height:1.35;list-style:decimal}' +
    '.rp .rp-lesson li{font-size:inherit;line-height:inherit;margin:0;padding:0}' +
    '.rp .rp-lesson li:last-child{color:var(--hazard,#F2B33D)}' +
    '.rp .rp-lesson h4{font-size:calc(var(--rp-fs,26px)*1.4);line-height:1.15;margin:0}' +
    '.rp-lesson .x{align-self:flex-start;margin-top:4px}' +
    '.rp-lane{display:flex;flex-direction:column;justify-content:center;gap:10px;padding:0 14px 0 4px;min-width:0;border-top:1px solid var(--rp-line)}' +
    '.rp-lane:first-child{border-top:none}' +
    '.rp-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}' +
    '.rp-name{font-weight:700;font-size:var(--rp-fs,26px)}' +
    '.rp-stat{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:calc(var(--rp-fs,26px)*.82);color:var(--rp-mute);white-space:normal;text-align:right;min-width:0}' +
    '.rp-stat.done{color:var(--rp-fg)}.rp-stat.fell{color:var(--rp-alarm)}' +
    '.rp-track{position:relative;height:calc(var(--rp-fs,26px)*1.2)}' +
    '.rp-rail{position:absolute;left:0;right:0;top:50%;height:4px;margin-top:-2px;background:var(--rp-line);border-radius:2px}' +
    '.rp-fill{position:absolute;left:0;top:50%;height:4px;margin-top:-2px;border-radius:2px;width:0}' +
    '.rp-tick{position:absolute;top:50%;width:2px;height:14px;margin:-7px 0 0 -1px;background:var(--rp-mute);opacity:.55;transition:opacity .2s,transform .2s}' +
    '.rp-tick.hit{opacity:1;transform:scaleY(1.25)}' +
    '.rp-bot{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:var(--rp-fg);box-shadow:0 0 0 3px rgba(0,0,0,.25)}' +
    '.rp-fin{position:absolute;top:50%;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;opacity:.28;border:2px solid currentColor;background:transparent;transition:opacity .25s}' +
    '.rp-fin.on{opacity:1;background:currentColor;animation:rp-pop .6s cubic-bezier(.2,1.6,.4,1)}' +
    '.rp-fall{position:absolute;top:50%;width:22px;height:22px;margin:-11px 0 0 -11px;transform:rotate(45deg);border:2px solid var(--rp-alarm);opacity:.28;transition:opacity .25s}' +
    '.rp-fall.on{opacity:1;background:var(--rp-alarm);animation:rp-pop2 .6s cubic-bezier(.2,1.6,.4,1)}' +
    '.rp-rank{position:absolute;top:100%;transform:translate(-50%,6px);font-family:var(--mono,"JetBrains Mono",monospace);font-size:calc(var(--rp-fs,26px)*.7);white-space:nowrap;opacity:0;transition:opacity .25s}' +
    '.rp-rank.on{opacity:1}' +
    '@keyframes rp-pop{0%{transform:scale(.3)}100%{transform:scale(1)}}' +
    '@keyframes rp-pop2{0%{transform:rotate(45deg) scale(.3)}100%{transform:rotate(45deg) scale(1)}}' +
    '.rp-ctl{display:flex;align-items:center;gap:16px;flex-wrap:wrap}' +
    '.rp-btn{font:inherit;font-size:calc(var(--rp-fs,26px)*.8);color:var(--rp-fg);background:var(--rp-panel);border:1px solid var(--rp-line);border-radius:8px;padding:6px 14px;cursor:pointer;line-height:1.2}' +
    '.rp-btn:hover{border-color:var(--rp-mute)}.rp-btn:focus-visible{outline:2px solid var(--hazard,#F2B33D);outline-offset:2px}' +
    '.rp-btn[aria-pressed="true"]{background:var(--rp-fg);color:var(--ink,#0E1724);border-color:var(--rp-fg)}' +
    '.rp-clock{font-family:var(--mono,"JetBrains Mono",monospace);font-size:calc(var(--rp-fs,26px)*.85);min-width:9.5em;font-variant-numeric:tabular-nums}' +
    '.rp-scrub{position:relative;flex:1;min-width:200px;height:34px;cursor:pointer;touch-action:none}' +
    '.rp-scrub .rp-rail{height:6px;margin-top:-3px}' +
    '.rp-scrub .rp-fill{height:6px;margin-top:-3px;background:var(--rp-fg)}' +
    '.rp-head2{position:absolute;top:50%;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;background:var(--rp-fg);box-shadow:0 0 0 4px rgba(0,0,0,.3)}' +
    '.rp-mk{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%}' +
    '.rp-mk.f{border-radius:2px;transform:rotate(45deg);background:var(--rp-alarm)}' +
    '.rp-note{font-size:calc(var(--rp-fs,26px)*.75);color:var(--rp-mute)}' +
    '@media (max-width:760px){.rp-top{flex-direction:column}.rp-vid{height:auto;width:100%}.rp-lanes{gap:14px}.rp-lane{border-top:none}}' +
    '@media (prefers-reduced-motion:reduce){.rp-fin.on,.rp-fall.on{animation:none}}';

  function injectCss() {
    if (document.getElementById('rp-css')) return;
    var s = el('style'); s.id = 'rp-css'; s.textContent = CSS; document.head.appendChild(s);
  }

  // Finish order among lanes that finished: 1st, 2nd ...
  var finishers = RACE.lanes.filter(function (l) { return l.done != null; }).sort(function (a, b) { return a.done - b.done; });
  var ORD = ['1st', '2nd', '3rd'];

  function mount(root) {
    if (root._rp) return root._rp;
    injectCss();
    root.classList.add('rp');
    var rate = parseFloat(root.getAttribute('data-rate') || '8');

    var top = el('div', 'rp-top');
    var vwrap = el('div', 'rp-vid');
    var v = el('video');
    v.src = root.getAttribute('data-src'); v.poster = root.getAttribute('data-poster') || '';
    v.muted = true; v.playsInline = true; v.preload = 'metadata'; v.setAttribute('data-race', '');
    v.setAttribute('aria-label', 'Three robots on the same patrol: no avoidance, hand-written planner, learned policy');
    vwrap.appendChild(v); top.appendChild(vwrap);

    var lanesBox = el('div', 'rp-lanes');
    var laneUI = RACE.lanes.map(function (L) {
      var row = el('div', 'rp-lane');
      var head = el('div', 'rp-head');
      head.appendChild(el('span', 'rp-name', L.name));
      var stat = el('span', 'rp-stat', 'Checkpoint 0 of ' + NCP); head.appendChild(stat);
      row.appendChild(head);
      var tr = el('div', 'rp-track'); tr.style.color = L.color;
      tr.appendChild(el('div', 'rp-rail'));
      var fill = el('div', 'rp-fill'); fill.style.background = L.color; tr.appendChild(fill);
      var ticks = L.cps.map(function (t) { var k = el('div', 'rp-tick'); k.style.left = pct(t); tr.appendChild(k); return k; });
      var fin = null, rank = null, fall = null;
      if (L.done != null) {
        fin = el('div', 'rp-fin'); fin.style.left = pct(L.done); fin.title = 'Finished ' + fmt(L.done); tr.appendChild(fin);
        rank = el('div', 'rp-rank', ORD[finishers.indexOf(L)] + ' · ' + fmt(L.done)); rank.style.left = pct(L.done); tr.appendChild(rank);
      }
      if (L.fell != null) { fall = el('div', 'rp-fall'); fall.style.left = pct(L.fell); fall.title = L.fellNote + ' ' + fmt(L.fell); tr.appendChild(fall); }
      var bot = el('div', 'rp-bot'); tr.appendChild(bot);
      row.appendChild(tr); lanesBox.appendChild(row);
      return { L: L, stat: stat, fill: fill, ticks: ticks, fin: fin, rank: rank, fall: fall, bot: bot };
    });
    var lesson = null, lessonHidden = false;
    if (RACE.lesson) {
      lesson = el('div', 'rp-lesson');
      lesson.setAttribute('role', 'note');
      lesson.innerHTML = '<span class="k">The lesson</span><h4></h4><ol></ol>';
      lesson.querySelector('h4').textContent = RACE.lesson.title;
      var ol = lesson.querySelector('ol');
      RACE.lesson.points.forEach(function (t) { ol.appendChild(el('li', null)).textContent = t; });
      var x = el('button', 'rp-btn x', 'Show the trackers'); x.type = 'button';
      x.addEventListener('click', function (e) { e.stopPropagation(); lessonHidden = true; lesson.classList.remove('on'); });
      lesson.appendChild(x);
      lanesBox.appendChild(lesson);
    }
    top.appendChild(lanesBox);
    root.appendChild(top);

    var ctl = el('div', 'rp-ctl');
    var play = el('button', 'rp-btn', 'Play'); play.type = 'button';
    var clock = el('span', 'rp-clock', '0:00.0 / ' + fmt(RACE.duration));
    var scrub = el('div', 'rp-scrub'); scrub.setAttribute('role', 'slider'); scrub.setAttribute('aria-label', 'Race time');
    scrub.setAttribute('aria-valuemin', '0'); scrub.setAttribute('aria-valuemax', String(RACE.duration)); scrub.tabIndex = 0;
    scrub.appendChild(el('div', 'rp-rail'));
    var sfill = el('div', 'rp-fill'); scrub.appendChild(sfill);
    RACE.lanes.forEach(function (L) {
      if (L.done != null) { var m = el('div', 'rp-mk'); m.style.left = pct(L.done); m.style.background = L.color; m.title = L.name + ' finished ' + fmt(L.done); scrub.appendChild(m); }
      if (L.fell != null) { var f = el('div', 'rp-mk f'); f.style.left = pct(L.fell); f.title = L.name + ' ' + L.fellNote + ' ' + fmt(L.fell); scrub.appendChild(f); }
    });
    var shead = el('div', 'rp-head2'); scrub.appendChild(shead);
    ctl.appendChild(play); ctl.appendChild(clock); ctl.appendChild(scrub);
    var speeds = [1, 4, 8, 16].map(function (r) {
      var b = el('button', 'rp-btn', r + '×'); b.type = 'button'; b.setAttribute('aria-pressed', String(r === rate));
      b.addEventListener('click', function (e) { e.stopPropagation(); setRate(r); }); ctl.appendChild(b); return { r: r, b: b };
    });
    var jump = el('button', 'rp-btn', 'Jump to the finish'); jump.type = 'button'; ctl.appendChild(jump);
    root.appendChild(ctl);
    if (root.hasAttribute('data-note')) root.appendChild(el('p', 'rp-note', root.getAttribute('data-note')));

    function setRate(r) { rate = r; v.playbackRate = r; speeds.forEach(function (s) { s.b.setAttribute('aria-pressed', String(s.r === r)); }); }
    v.addEventListener('loadedmetadata', function () { v.playbackRate = rate; });
    v.addEventListener('play', function () { v.playbackRate = rate; play.textContent = 'Pause'; });
    v.addEventListener('pause', function () { play.textContent = 'Play'; });
    function toggle(e) { if (e) e.stopPropagation(); if (v.paused) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else v.pause(); }
    play.addEventListener('click', toggle);
    v.addEventListener('click', toggle);
    jump.addEventListener('click', function (e) { e.stopPropagation(); v.currentTime = RACE.jumpTo; render(RACE.jumpTo); if (v.paused) toggle(); });

    function render(t) {
      if (lesson) { if (t < RACE.lesson.at) lessonHidden = false; lesson.classList.toggle('on', t >= RACE.lesson.at && !lessonHidden); }
      clock.textContent = fmt(t) + ' / ' + fmt(RACE.duration);
      sfill.style.width = pct(t); shead.style.left = pct(t);
      scrub.setAttribute('aria-valuenow', t.toFixed(1)); scrub.setAttribute('aria-valuetext', fmt0(t));
      laneUI.forEach(function (u) {
        var L = u.L, n = 0; L.cps.forEach(function (c, i) { var h = t >= c; if (h) n++; u.ticks[i].classList.toggle('hit', h); });
        var end = Math.min(L.done != null ? L.done : Infinity, L.fell != null ? L.fell : Infinity);
        var pos = Math.min(t, end, RACE.duration);
        u.fill.style.width = pct(pos); u.bot.style.left = pct(pos);
        var isDone = L.done != null && t >= L.done, isFell = L.fell != null && t >= L.fell;
        if (u.fin) u.fin.classList.toggle('on', isDone);
        if (u.rank) u.rank.classList.toggle('on', isDone);
        if (u.fall) u.fall.classList.toggle('on', isFell);
        u.bot.style.opacity = (isDone || isFell) ? '0' : '1';
        var txt, cls = 'rp-stat';
        if (isFell && !isDone) { txt = 'Knocked over at ' + fmt0(L.fell) + ' · ' + n + ' of ' + NCP; cls += ' fell'; }
        else if (isDone && isFell) { txt = ORD[finishers.indexOf(L)] + ', done ' + fmt0(L.done) + ' · knocked over parked, by a late walker'; cls += ' fell'; }
        else if (isDone) { txt = ORD[finishers.indexOf(L)] + ', done at ' + fmt0(L.done); cls += ' done'; }
        else if (n === NCP) txt = 'All 8 checkpoints · heading home';
        else txt = 'Checkpoint ' + n + ' of ' + NCP;
        u.stat.className = cls; u.stat.textContent = txt;
      });
    }

    var raf = 0;
    function loop() { render(v.currentTime || 0); raf = v.paused ? 0 : requestAnimationFrame(loop); }
    v.addEventListener('play', function () { if (!raf) raf = requestAnimationFrame(loop); });
    v.addEventListener('seeked', function () { render(v.currentTime); });
    v.addEventListener('timeupdate', function () { if (!raf) render(v.currentTime); });

    function seekFromEvent(e) {
      var r = scrub.getBoundingClientRect(); var x = (e.clientX - r.left) / r.width;
      var t = Math.max(0, Math.min(1, x)) * RACE.duration; v.currentTime = t; render(t);
    }
    var dragging = false;
    scrub.addEventListener('pointerdown', function (e) { e.stopPropagation(); dragging = true; try { scrub.setPointerCapture(e.pointerId); } catch (_) {} seekFromEvent(e); });
    scrub.addEventListener('pointermove', function (e) { if (dragging) seekFromEvent(e); });
    scrub.addEventListener('pointerup', function () { dragging = false; });
    scrub.addEventListener('click', function (e) { e.stopPropagation(); });
    scrub.addEventListener('keydown', function (e) {
      var d = e.key === 'ArrowRight' ? 5 : e.key === 'ArrowLeft' ? -5 : 0;
      if (d) { e.preventDefault(); e.stopPropagation(); v.currentTime = Math.max(0, Math.min(RACE.duration, v.currentTime + d)); }
    });
    [ctl, lanesBox].forEach(function (n) { n.addEventListener('click', function (e) { e.stopPropagation(); }); });

    render(0);
    root._rp = { video: v, setRate: setRate, render: render };
    return root._rp;
  }

  window.RacePlayer = {
    data: RACE,
    mount: mount,
    mountAll: function (sel) { [].forEach.call(document.querySelectorAll(sel || '.race'), mount); }
  };
})();
