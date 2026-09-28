/* Cultural Data Analysis – shared course bar, keyboard shortcuts, presentation mode and deep links.
 *
 * Include at the end of <body> with data attributes:
 *   <script src="../../assets/course.js"
 *     data-week="1" data-title="CPU simulator"
 *     data-guide="guide.html"            (optional: link to the tool's guide page)
 *     data-step="#btn-step"              (optional: button for Space and ?steps=N; "text:Step" matches a button label)
 *     data-reset="#btn-reset"            (optional: button for R)
 *     data-run="#btn-run"                (optional: button for Enter)
 *     data-step-label="New image"        (optional: what Space does, for the help panel)
 *     data-note="Illustrative data"      (optional: chip in the bar)
 *     data-fit="height"                  (optional: page fills the window, keep it that way in presentation mode)
 *     data-min-size="1000x680"           (optional: smallest size the page needs; presentation mode zooms no further than fits)
 *     data-help="Line one|Line two"      (optional: extra lines in the help panel)
 *     data-guide-page="1"></script>      (on a guide page itself: links back to the tool)
 *
 * URL parameters understood on every tool: ?present=1 (start in presentation mode), ?steps=N (press Step N times).
 * Tool-specific parameters are handled in each tool's own script.
 */
(function () {
  'use strict';
  var me = document.currentScript;
  var cfg = me ? me.dataset : {};
  var root = new URL('../', me ? me.src : location.href).href;
  var params = new URLSearchParams(location.search);
  var WEEKS = { 0: 'Python Bootcamp', 1: 'Python', 2: 'Data', 3: 'Structured Data and Machine Learning', 4: 'Cultural Data Forms: Text' };

  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (html != null) e.innerHTML = html;
    return e;
  }
  function find(sel) {
    if (!sel) return null;
    if (sel.indexOf('text:') === 0) {
      // "text:Train|Pause" matches the first button whose label contains any of the words
      var ts = sel.slice(5).toLowerCase().split('|');
      var bs = document.querySelectorAll('button:not(.cda-topbar button)');
      for (var i = 0; i < bs.length; i++) {
        var label = bs[i].textContent.toLowerCase();
        for (var j = 0; j < ts.length; j++) if (label.indexOf(ts[j]) !== -1) return bs[i];
      }
      return null;
    }
    return document.querySelector(sel);
  }
  function press(sel) {
    var b = find(sel);
    if (b && !b.disabled) { b.click(); flash(b); return true; }
    return false;
  }
  function flash(b) {
    b.style.outline = '3px solid #F0C4CE';
    setTimeout(function () { b.style.outline = ''; }, 180);
  }
  var toastEl;
  function toast(msg) {
    if (!toastEl) { toastEl = el('div', { 'class': 'cda-toast', role: 'status' }); document.body.appendChild(toastEl); }
    toastEl.textContent = msg; toastEl.classList.add('on');
    clearTimeout(toast.t); toast.t = setTimeout(function () { toastEl.classList.remove('on'); }, 1200);
  }

  // ---------- the bar
  var week = cfg.week;
  var bar = el('div', { 'class': 'cda-topbar', role: 'banner' });
  var crumbs = '<a class="cda-course" href="' + root + '">Cultural Data Analysis</a>';
  if (week != null && week !== '') {
    crumbs += '<span class="cda-sep">/</span><a href="' + root + 'code-along/week' + week + '/">Week ' + week + (WEEKS[week] ? ': ' + WEEKS[week] : '') + '</a>';
  }
  if (cfg.title) crumbs += '<span class="cda-sep">/</span><span class="cda-tool">' + cfg.title + '</span>';
  if (cfg.note) crumbs += ' <span class="cda-chip" title="The numbers on this page are made up to show the idea; they do not come from a real model.">' + cfg.note + '</span>';
  var nav = '';
  var shortUrl = (location.host + location.pathname).replace(/index\.html$/, '');
  nav += '<span class="cda-url" title="Address of this page">' + shortUrl + '</span>';
  if (cfg.guidePage) nav += '<a class="cda-primary" href="./">Open the tool</a>';
  else if (cfg.guide) nav += '<a href="' + cfg.guide + '">Guide</a>';
  if (week != null && week !== '') nav += '<a href="' + root + 'code-along/week' + week + '/">Code along</a>';
  nav += '<a href="' + root + '#tools">All tools</a>';
  if (!cfg.guidePage) nav += '<button type="button" class="cda-keep" data-cda="present" title="Presentation mode (P)">Present</button>';
  nav += '<button type="button" class="cda-keep" data-cda="help" title="Keyboard shortcuts (?)" aria-label="Help">?</button>';
  bar.innerHTML = '<div class="cda-crumbs" role="navigation" aria-label="Course">' + crumbs + '</div><div class="cda-nav">' + nav + '</div>';
  document.body.insertBefore(bar, document.body.firstChild);
  document.body.classList.add('cda-has-bar');
  if (cfg.fit === 'height') document.body.classList.add('cda-fit');

  // ---------- help panel
  var rows = [];
  if (cfg.step) rows.push(['Space', cfg.stepLabel || 'Step: one step forward']);
  if (cfg.run) rows.push(['Enter', 'Run / pause']);
  if (cfg.reset) rows.push(['R', 'Reset']);
  if (!cfg.guidePage) {
    rows.push(['P', 'Presentation mode on/off: larger, shows the page address']);
    rows.push(['+ / &minus;', 'Larger / smaller in presentation mode']);
  }
  if (cfg.guide) rows.push(['G', 'Open the guide']);
  rows.push(['?', 'This help']);
  rows.push(['Esc', 'Close']);
  var extra = (cfg.help || '').split('|').filter(Boolean).map(function (s) { return '<p>' + s + '</p>'; }).join('');
  var deep = '';
  if (cfg.step) deep = '<p>Link straight to a state: add <kbd>?steps=5</kbd> to the address to start five steps in, or <kbd>?present=1</kbd> to open in presentation mode.</p>';
  var back = el('div', { 'class': 'cda-help-back', hidden: '' });
  back.innerHTML = '<div class="cda-help" role="dialog" aria-modal="true" aria-labelledby="cda-help-h">' +
    '<button type="button" class="cda-close" aria-label="Close">&times;</button>' +
    '<h2 id="cda-help-h">Keyboard shortcuts</h2><table>' +
    rows.map(function (r) { return '<tr><td><kbd>' + r[0] + '</kbd></td><td>' + r[1] + '</td></tr>'; }).join('') +
    '</table>' + extra + deep +
    (cfg.guide ? '<p>The <a href="' + cfg.guide + '">guide</a> explains every part of this tool, with walkthroughs for class.</p>' : '') +
    '</div>';
  document.body.appendChild(back);
  var lastFocus = null;
  function help(open) {
    if (open) { lastFocus = document.activeElement; back.hidden = false; back.querySelector('.cda-close').focus(); }
    else { back.hidden = true; if (lastFocus && lastFocus.focus) lastFocus.focus(); }
  }
  back.addEventListener('click', function (e) { if (e.target === back || e.target.classList.contains('cda-close')) help(false); });

  // ---------- presentation mode
  var zoom = 1.3;
  try { var z = parseFloat(localStorage.getItem('cda-zoom')); if (z >= 1 && z <= 2.5) zoom = z; } catch (e) { /* storage unavailable */ }
  // pages that must fit the window (data-min-size="1000x680") never zoom further than fits
  var minSize = (cfg.minSize || '').split('x').map(Number);
  function effective() {
    if (!(minSize[0] > 0 && minSize[1] > 0)) return zoom;
    var fit = Math.min(innerWidth / minSize[0], innerHeight / minSize[1]);
    return Math.max(1, Math.min(zoom, Math.floor(fit * 20) / 20));
  }
  function setZoom(z, keep) {
    zoom = Math.round(Math.min(2.5, Math.max(1, z)) * 10) / 10;
    document.documentElement.style.setProperty('--cda-zoom', effective());
    if (!keep) { try { localStorage.setItem('cda-zoom', zoom); } catch (e) { /* ignore */ } }
    window.dispatchEvent(new Event('resize'));
  }
  function present(on) {
    document.documentElement.classList.toggle('cda-present', on);
    setZoom(zoom, true);
    toast(on ? 'Presentation mode · ' + Math.round(effective() * 100) + '% · P to leave' : 'Presentation mode off');
  }
  window.addEventListener('resize', function (e) {
    if (e.isTrusted) document.documentElement.style.setProperty('--cda-zoom', effective());
  });
  setZoom(zoom, true);
  var pz = parseFloat(params.get('present'));
  if (params.has('present') && params.get('present') !== '0') { if (pz > 1) setZoom(pz); document.documentElement.classList.add('cda-present'); }

  bar.addEventListener('click', function (e) {
    var b = e.target.closest('[data-cda]');
    if (!b) return;
    if (b.dataset.cda === 'help') help(true);
    if (b.dataset.cda === 'present') present(!document.documentElement.classList.contains('cda-present'));
  });

  // ---------- keyboard
  // after a mouse click, let go of the button so that Space means "Step" again
  document.addEventListener('mouseup', function (e) {
    var b = e.target.closest && e.target.closest('button');
    if (b) setTimeout(function () { b.blur(); }, 0);
  });
  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target;
    var typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
    if (!back.hidden) { if (e.key === 'Escape' || e.key === '?') { e.preventDefault(); help(false); } return; }
    if (typing) { if (e.key === 'Escape') t.blur(); return; }
    var k = e.key;
    if (k === '?') { e.preventDefault(); help(true); }
    else if (k === ' ' && cfg.step) {
      if (t && t.tagName === 'BUTTON' && t !== document.body) { /* let the focused button handle Space */ return; }
      e.preventDefault(); press(cfg.step);
    }
    else if (k === 'Enter' && cfg.run && !(t && /^(BUTTON|A)$/.test(t.tagName))) { e.preventDefault(); press(cfg.run); }
    else if ((k === 'r' || k === 'R') && cfg.reset) { e.preventDefault(); press(cfg.reset); }
    else if ((k === 'p' || k === 'P') && !cfg.guidePage) { e.preventDefault(); present(!document.documentElement.classList.contains('cda-present')); }
    else if ((k === '+' || k === '=') && document.documentElement.classList.contains('cda-present')) { setZoom(zoom + 0.1); toast(Math.round(effective() * 100) + '%'); }
    else if ((k === '-' || k === '_') && document.documentElement.classList.contains('cda-present')) { setZoom(zoom - 0.1); toast(Math.round(effective() * 100) + '%'); }
    else if ((k === 'g' || k === 'G') && cfg.guide) { location.href = cfg.guide; }
  });

  // ---------- ?steps=N
  window.CDA = {
    params: params,
    press: press,
    toast: toast,
    // press Step n times, waiting between presses so the page can redraw
    steps: function (n, delay) {
      n = Math.min(parseInt(n, 10) || 0, 500);
      var i = 0;
      (function next() {
        if (i++ >= n) return;
        press(cfg.step);
        setTimeout(next, delay || 40);
      })();
    }
  };
  if (cfg.step && params.has('steps')) {
    window.addEventListener('load', function () {
      // tools may prepare state first (e.g. load a program); they set CDA.ready to a promise
      Promise.resolve(window.CDA.ready).then(function () { setTimeout(function () { window.CDA.steps(params.get('steps')); }, 250); });
    });
  }
})();
