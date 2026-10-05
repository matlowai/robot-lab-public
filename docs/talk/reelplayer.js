/* Reel player: plays media/opening_reel.mp4 with a chapter timeline and optional stop points.
   Stop points come from reel/chapters.js (window.REEL_CHAPTERS, written by reel/render.mjs from the same
   TIMELINE that renders the video), so the markers always match the cut.
   Usage: <div class="reelp" data-src="media/opening_reel.mp4" data-poster="media/opening_reel_still.jpg"></div>
   then ReelPlayer.mountAll(). "Pause at key moments" stops at each marked moment with a card saying why;
   Play / Space / → continues. The toggle is remembered per browser. Without chapters.js it degrades to a
   plain player. */
(function () {
  var KEY = 'reelPauseAtStops';
  function getPref() { try { var v = localStorage.getItem(KEY); return v == null ? true : v === '1'; } catch (e) { return true; } }
  function setPref(on) { try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (e) {} }
  function fmt(t) { t = Math.max(0, t || 0); var m = Math.floor(t / 60), s = Math.floor(t - m * 60); return m + ':' + (s < 10 ? '0' : '') + s; }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  var CSS = '' +
    '.rlp{--f:var(--paper,#EEF1F4);--m:var(--muted,#9AA8BA);--l:var(--line,#26364D);--p:var(--panel,#152234);--h:var(--hazard,#F2B33D);' +
    'display:flex;flex-direction:column;gap:12px;color:var(--f);width:100%;outline:none}' +
    '.rlp-box{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:12px;overflow:hidden;border:1px solid var(--l)}' +
    '.rlp-box video{display:block;width:100%;height:100%;object-fit:contain;background:#000;cursor:pointer;border:none;border-radius:0}' +
    '.rlp-card{display:none;gap:20px;align-items:center;padding:16px 22px;border-radius:12px;background:var(--p);border:1px solid var(--h)}' +
    '.rlp-card.on{display:flex;animation:rlp-in .3s ease-out}' +
    '@keyframes rlp-in{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}' +
    '.rlp-card .go{flex:none;width:calc(var(--rlp-fs,18px)*3.4);height:calc(var(--rlp-fs,18px)*3.4);border-radius:50%;border:none;background:var(--h);color:#0E1724;cursor:pointer;display:grid;place-items:center}' +
    '.rlp-card .go svg{width:42%;height:42%;margin-left:8%}' +
    '.rlp-card .tx{display:flex;flex-direction:column;gap:4px;min-width:0}' +
    '.rlp-card .k{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:calc(var(--rlp-fs,18px)*.72);letter-spacing:.14em;text-transform:uppercase;color:var(--h)}' +
    '.rlp-card .t{font-weight:700;font-size:calc(var(--rlp-fs,18px)*1.35);line-height:1.15;color:var(--f)}' +
    '.rlp-card .w{font-size:calc(var(--rlp-fs,18px)*.95);color:var(--f);opacity:.85;line-height:1.4}' +
    '.rlp-card .h{font-size:calc(var(--rlp-fs,18px)*.75);color:var(--m)}' +
    '.rlp-big{position:absolute;inset:0;display:grid;place-items:center;background:rgba(0,0,0,.25);border:none;cursor:pointer;opacity:0;pointer-events:none;transition:opacity .2s}' +
    '.rlp-big.on{opacity:1;pointer-events:auto}' +
    '.rlp-big span{width:calc(var(--rlp-fs,18px)*5);height:calc(var(--rlp-fs,18px)*5);border-radius:50%;background:var(--h);display:grid;place-items:center}' +
    '.rlp-big svg{width:40%;height:40%;margin-left:8%}' +
    '.rlp-bar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:var(--rlp-fs,18px)}' +
    '.rlp-btn{font:inherit;font-size:calc(var(--rlp-fs,18px)*.85);color:var(--f);background:var(--p);border:1px solid var(--l);border-radius:8px;padding:6px 14px;cursor:pointer;line-height:1.2}' +
    '.rlp-btn:hover{border-color:var(--m)}.rlp-btn:focus-visible,.rlp-sw:focus-within{outline:2px solid var(--h);outline-offset:2px}' +
    '.rlp-clock{font-family:var(--mono,"JetBrains Mono",monospace);font-size:calc(var(--rlp-fs,18px)*.85);font-variant-numeric:tabular-nums;min-width:6.5em}' +
    '.rlp-tl{position:relative;flex:1;min-width:220px;height:calc(var(--rlp-fs,18px)*2.6);cursor:pointer;touch-action:none}' +
    '.rlp-seg{position:absolute;top:40%;height:6px;margin-top:-3px;background:var(--l);border-radius:3px}' +
    '.rlp-seg.cur{background:color-mix(in srgb,var(--h) 45%,var(--l))}' +
    '.rlp-fill{position:absolute;left:0;top:40%;height:6px;margin-top:-3px;background:var(--f);border-radius:3px;width:0}' +
    '.rlp-lab{position:absolute;top:62%;font-family:var(--mono,"JetBrains Mono",monospace);font-size:calc(var(--rlp-fs,18px)*.62);color:var(--m);white-space:nowrap;transform:translateX(2px);overflow:hidden}' +
    '.rlp-lab.cur{color:var(--h)}' +
    '.rlp-stop{position:absolute;top:40%;width:12px;height:12px;margin:-6px 0 0 -6px;transform:rotate(45deg);background:var(--h);border:2px solid var(--ink,#0E1724);opacity:.35;transition:opacity .2s}' +
    '.rlp.pausing .rlp-stop{opacity:1}.rlp-stop.done{opacity:.25!important}' +
    '.rlp-head{position:absolute;top:40%;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;background:var(--f);box-shadow:0 0 0 3px rgba(0,0,0,.35)}' +
    '.rlp-sw{display:inline-flex;align-items:center;gap:10px;cursor:pointer;font-size:calc(var(--rlp-fs,18px)*.85);user-select:none}' +
    '.rlp-sw input{position:absolute;opacity:0;width:1px;height:1px}' +
    '.rlp-sw .tr{width:2.4em;height:1.3em;border-radius:1em;background:var(--l);position:relative;transition:background .2s;flex:none}' +
    '.rlp-sw .tr::after{content:"";position:absolute;top:.15em;left:.15em;width:1em;height:1em;border-radius:50%;background:var(--f);transition:transform .2s}' +
    '.rlp-sw input:checked + .tr{background:var(--h)}.rlp-sw input:checked + .tr::after{transform:translateX(1.1em)}' +
    '@media (prefers-reduced-motion:reduce){.rlp-card.on{animation:none}.rlp-big,.rlp-sw .tr::after{transition:none}}';

  var PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l16 9-16 9z" fill="#0E1724"/></svg>';

  function injectCss() { if (document.getElementById('rlp-css')) return; var s = el('style'); s.id = 'rlp-css'; s.textContent = CSS; document.head.appendChild(s); }

  var instances = [];

  function mount(root) {
    if (root._rlp) return root._rlp;
    injectCss();
    var data = window.REEL_CHAPTERS || null;
    root.classList.add('rlp'); root.tabIndex = 0;
    root.setAttribute('aria-label', 'Opening reel player');

    var box = el('div', 'rlp-box');
    var v = el('video'); v.src = root.getAttribute('data-src'); v.poster = root.getAttribute('data-poster') || '';
    v.playsInline = true; v.preload = 'auto'; v.setAttribute('data-reel', '');
    box.appendChild(v);
    var big = el('button', 'rlp-big', '<span>' + PLAY + '</span>'); big.type = 'button'; big.setAttribute('aria-label', 'Play the reel');
    box.appendChild(big);
    var card = el('div', 'rlp-card'); card.setAttribute('role', 'status'); card.setAttribute('aria-live', 'polite');
    card.innerHTML = '<button type="button" class="go" aria-label="Continue">' + PLAY + '</button>' +
      '<div class="tx"><span class="k">Paused here on purpose</span><span class="t"></span><span class="w"></span>' +
      '<span class="h">Press Play, Space or → to continue · turn off "Pause at key moments" to play straight through</span></div>';
    root.appendChild(box);
    root.appendChild(card);

    var bar = el('div', 'rlp-bar');
    var play = el('button', 'rlp-btn', 'Play'); play.type = 'button';
    var clock = el('span', 'rlp-clock', '0:00 / 0:00');
    var tl = el('div', 'rlp-tl'); tl.setAttribute('role', 'slider'); tl.setAttribute('aria-label', 'Reel position'); tl.tabIndex = 0;
    var fill = el('div', 'rlp-fill'), head = el('div', 'rlp-head');
    var sw = el('label', 'rlp-sw', '<input type="checkbox"><span class="tr" aria-hidden="true"></span><span>Pause at key moments</span>');
    var cb = sw.querySelector('input'); cb.checked = getPref();
    var mute = el('button', 'rlp-btn', 'Sound on'); mute.type = 'button';
    bar.appendChild(play); bar.appendChild(clock); bar.appendChild(tl);
    if (data && data.stops && data.stops.length) bar.appendChild(sw);
    bar.appendChild(mute);
    root.appendChild(bar);

    var D = (data && data.duration) || 0, chapters = (data && data.chapters) || [], stops = ((data && data.stops) || []).slice().sort(function (a, b) { return a.t - b.t; });
    var segEls = [], labEls = [], stopEls = [];
    function layout() {
      [].slice.call(tl.children).forEach(function (c) { tl.removeChild(c); });
      segEls = []; labEls = []; stopEls = [];
      if (!D) { tl.appendChild(el('div', 'rlp-seg')).style.cssText = 'left:0;right:0'; }
      chapters.forEach(function (c, i) {
        var a = c.t / D, b = (i + 1 < chapters.length ? chapters[i + 1].t : D) / D;
        var s = el('div', 'rlp-seg'); s.style.left = (a * 100) + '%'; s.style.width = 'calc(' + ((b - a) * 100) + '% - 3px)'; tl.appendChild(s); segEls.push(s);
        var l = el('span', 'rlp-lab', c.label); l.style.left = (a * 100) + '%'; l.style.maxWidth = 'calc(' + ((b - a) * 100) + '% - 6px)'; l.title = c.label; tl.appendChild(l); labEls.push(l);
      });
      tl.appendChild(fill);
      stops.forEach(function (s) { var d = el('div', 'rlp-stop'); d.style.left = (s.t / D * 100) + '%'; d.title = 'Stop: ' + s.title; tl.appendChild(d); stopEls.push(d); });
      tl.appendChild(head);
    }
    v.addEventListener('loadedmetadata', function () { if (!D) D = v.duration; layout(); render(); });
    if (D) layout();

    var armed = 0;          // index of the next stop that may fire
    var atStop = -1;        // index of the stop we are paused on, or -1
    function rearm(t) { armed = 0; while (armed < stops.length && stops[armed].t <= t + 0.05) armed++; stopEls.forEach(function (d, i) { d.classList.toggle('done', i < armed); }); }
    function pausing() { return cb.checked && stops.length > 0; }
    root.classList.toggle('pausing', pausing());

    function showCard(i) {
      atStop = i; var s = stops[i];
      card.querySelector('.t').textContent = s.title; card.querySelector('.w').textContent = s.why || '';
      card.classList.add('on');
    }
    function hideCard() { atStop = -1; card.classList.remove('on'); }
    function doPlay() { hideCard(); big.classList.remove('on'); var p = v.play(); if (p && p.catch) p.catch(function () { big.classList.add('on'); }); }
    var selfPause = false; // true while a pause comes from this player (button, click, stop point, slide change)
    function toggle() { if (v.paused) doPlay(); else { selfPause = true; v.pause(); } }

    function render() {
      var t = v.currentTime || 0, d = D || v.duration || 1;
      clock.textContent = fmt(t) + ' / ' + fmt(d);
      fill.style.width = (t / d * 100) + '%'; head.style.left = (t / d * 100) + '%';
      tl.setAttribute('aria-valuenow', t.toFixed(1)); tl.setAttribute('aria-valuetext', fmt(t));
      var ci = -1; chapters.forEach(function (c, i) { if (t >= c.t) ci = i; });
      segEls.forEach(function (s, i) { s.classList.toggle('cur', i === ci); });
      labEls.forEach(function (s, i) { s.classList.toggle('cur', i === ci); });
    }
    var raf = 0;
    function loop() {
      var t = v.currentTime;
      if (pausing() && armed < stops.length && t >= stops[armed].t) {
        var i = armed; armed++; selfPause = true; v.pause(); v.currentTime = stops[i].t; stopEls[i] && stopEls[i].classList.add('done'); showCard(i);
      } else if (!pausing()) { while (armed < stops.length && stops[armed].t <= t) { stopEls[armed] && stopEls[armed].classList.add('done'); armed++; } }
      render();
      raf = v.paused ? 0 : requestAnimationFrame(loop);
    }
    v.addEventListener('play', function () { play.textContent = 'Pause'; hideCard(); big.classList.remove('on'); if (!raf) raf = requestAnimationFrame(loop); });
    // A pause the player didn't ask for (the browser or OS paused it): show the big Play button so it
    // reads as paused, not as a glitch.
    v.addEventListener('pause', function () { play.textContent = 'Play'; if (!selfPause && !v.ended && atStop < 0) big.classList.add('on'); selfPause = false; });
    v.addEventListener('ended', function () { play.textContent = 'Replay'; render(); });
    v.addEventListener('seeked', render);
    v.addEventListener('timeupdate', function () { if (!raf) render(); });
    v.addEventListener('volumechange', function () { mute.textContent = v.muted ? 'Sound off' : 'Sound on'; });

    play.addEventListener('click', function (e) { e.stopPropagation(); if (v.ended) { v.currentTime = 0; rearm(0); } toggle(); });
    v.addEventListener('click', function (e) { e.stopPropagation(); toggle(); });
    big.addEventListener('click', function (e) { e.stopPropagation(); doPlay(); });
    card.querySelector('.go').addEventListener('click', function (e) { e.stopPropagation(); doPlay(); });
    mute.addEventListener('click', function (e) { e.stopPropagation(); v.muted = !v.muted; });
    cb.addEventListener('change', function () { setPref(cb.checked); root.classList.toggle('pausing', pausing()); if (!cb.checked && atStop >= 0) doPlay(); });
    sw.addEventListener('click', function (e) { e.stopPropagation(); });

    function seekTo(e) { var r = tl.getBoundingClientRect(); var t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (D || v.duration || 0); hideCard(); v.currentTime = t; rearm(t); render(); }
    var drag = false;
    tl.addEventListener('pointerdown', function (e) { e.stopPropagation(); drag = true; try { tl.setPointerCapture(e.pointerId); } catch (_) {} seekTo(e); });
    tl.addEventListener('pointermove', function (e) { if (drag) seekTo(e); });
    tl.addEventListener('pointerup', function () { drag = false; });
    tl.addEventListener('click', function (e) { e.stopPropagation(); });
    bar.addEventListener('click', function (e) { e.stopPropagation(); });

    // Keys when the player itself has focus (the deck calls advance() for its own keys).
    root.addEventListener('keydown', function (e) {
      if (e.target === cb) return;
      if (e.key === ' ' || e.key === 'k') { e.preventDefault(); e.stopPropagation(); toggle(); }
      else if (e.key === 'ArrowRight' && atStop >= 0) { e.preventDefault(); e.stopPropagation(); doPlay(); }
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { if (e.target === tl) { e.preventDefault(); e.stopPropagation(); var t = Math.max(0, v.currentTime + (e.key === 'ArrowLeft' ? -5 : 5)); v.currentTime = t; rearm(t); } }
    });

    var api = {
      video: v, root: root,
      // Deck hook: returns true when the key press was used to continue the reel.
      advance: function () { if (atStop >= 0 || (v.paused && !v.ended && v.currentTime > 0.1)) { doPlay(); return true; } return false; },
      start: function () { hideCard(); v.currentTime = 0; rearm(0); render(); doPlay(); },
      stop: function () { selfPause = true; v.pause(); hideCard(); }
    };
    root._rlp = api; instances.push(api);
    rearm(0); render(); big.classList.add('on');
    return api;
  }

  window.ReelPlayer = {
    mount: mount,
    mountAll: function (sel) { [].forEach.call(document.querySelectorAll(sel || '.reelp'), mount); },
    instances: instances
  };
})();
