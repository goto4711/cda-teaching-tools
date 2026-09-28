// The same loop, for i in 0, 1, 2: print it, in four languages side by side.
// Every Step is one phase of the loop (set up, test, body, next), highlighted in all four languages at once.

const LANGS = {
    assembly: {
        name: 'Assembly', color: 'var(--lang-assembly)',
        lines: [
            [['    MOV ECX, 0', 'init'], ['   ; counter = 0', 'c']],
            [['loop_start:', '']],
            [['    CMP ECX, 3', 'test end'], ['   ; compare with 3', 'c']],
            [['    JGE loop_end', 'test end'], [' ; 3 or more: leave', 'c']],
            [['    PUSH ECX', 'body'], ['     ; save counter', 'c']],
            [['    CALL print', 'body'], ['   ; print it', 'c']],
            [['    POP ECX', 'body'], ['      ; restore it', 'c']],
            [['    INC ECX', 'next'], ['      ; counter + 1', 'c']],
            [['    JMP loop_start', 'next'], [' ; back to test', 'c']],
            [['loop_end:', 'end']],
        ],
        // machine instructions executed in each phase
        cost: { init: 1, test: 2, body: 3, next: 2, end: 2 },
        says: {
            init: 'Put 0 in register ECX.',
            test: i => `Compare ECX (${i}) with 3; not yet 3, so no jump.`,
            body: i => `Save ECX, call print, restore ECX. Prints ${i}.`,
            next: i => `ECX becomes ${i + 1}; jump back to loop_start.`,
            end: 'Compare ECX (3) with 3; now the jump to loop_end is taken.',
        }
    },
    python: {
        name: 'Python', color: 'var(--lang-python)',
        lines: [
            [['for ', 'kw'], ['i', 'init next'], [' in ', 'kw'], ['range(3)', 'init test end'], [':', '']],
            [['    print(f"Iteration {i}")', 'body']],
        ],
        cost: { init: 1, test: 1, body: 1, next: 1, end: 1 },
        says: {
            init: 'range(3) makes 0, 1, 2; i takes the first, 0.',
            test: i => `range(3) still has a value (${i}), so the body runs.`,
            body: i => `Prints Iteration ${i}.`,
            next: i => `i takes the next value from range(3): ${i + 1 < 3 ? i + 1 : 'none left'}.`,
            end: 'range(3) has no values left: the loop ends.',
        }
    },
    javascript: {
        name: 'JavaScript', color: 'var(--lang-javascript)',
        lines: [
            [['for (', 'kw'], ['let i = 0', 'init'], ['; ', ''], ['i < 3', 'test end'], ['; ', ''], ['i++', 'next'], [') {', '']],
            [['    console.log(`Iteration ${i}`);', 'body']],
            [['}', '']],
        ],
        cost: { init: 1, test: 1, body: 1, next: 1, end: 1 },
        says: {
            init: 'let i = 0 runs once.',
            test: i => `i < 3 is true (${i} < 3).`,
            body: i => `Logs Iteration ${i}.`,
            next: i => `i++ makes i ${i + 1}.`,
            end: 'i < 3 is false (3 < 3): the loop ends.',
        }
    },
    cpp: {
        name: 'C++', color: 'var(--lang-cpp)',
        lines: [
            [['for (', 'kw'], ['int i = 0', 'init'], ['; ', ''], ['i < 3', 'test end'], ['; ', ''], ['i++', 'next'], [') {', '']],
            [['    std::cout << "Iteration "', 'body']],
            [['              << i << std::endl;', 'body']],
            [['}', '']],
        ],
        cost: { init: 1, test: 1, body: 1, next: 1, end: 1 },
        says: {
            init: 'int i = 0: i is declared as a whole number and set to 0.',
            test: i => `i < 3 is true (${i} < 3).`,
            body: i => `Prints Iteration ${i}.`,
            next: i => `i++ makes i ${i + 1}.`,
            end: 'i < 3 is false (3 < 3): the loop ends.',
        }
    }
};
const ORDER = ['assembly', 'python', 'javascript', 'cpp'];
const PHASE_NAMES = { init: 'Set up', test: 'Test', body: 'Body', next: 'Next', end: 'Test fails: end' };

// the whole run as a list of phases
const PLAN = [{ phase: 'init', i: 0 }];
for (let i = 0; i < 3; i++) PLAN.push({ phase: 'test', i }, { phase: 'body', i }, { phase: 'next', i });
PLAN.push({ phase: 'end', i: 3 });

// ?lang=python puts that language first and outlines it
const focusLang = (function () {
    const l = new URLSearchParams(location.search).get('lang');
    return ORDER.includes(l) ? l : null;
})();
const order = focusLang ? [focusLang].concat(ORDER.filter(k => k !== focusLang)) : ORDER;

let pos = -1;          // index into PLAN of the phase just shown
let timer = null;

const grid = document.getElementById('code-grid');
const phaseBar = document.getElementById('phase-bar');
const narration = document.getElementById('narration');
const outputLog = document.getElementById('output-log');
const outputContainer = document.getElementById('output-container');
const iterationContainer = document.getElementById('iteration-container');
const circles = document.querySelectorAll('.circle');
const btnExecute = document.getElementById('btn-execute');
const btnStep = document.getElementById('btn-step');
const btnReset = document.getElementById('btn-reset');

const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function build() {
    grid.innerHTML = order.map(k => {
        const L = LANGS[k];
        const code = L.lines.map(line => '<div class="code-line">' + line.map(([text, tags]) =>
            `<span class="seg${tags === 'c' ? ' cmt' : ''}" data-tags="${tags}">${esc(text)}</span>`).join('') + '</div>').join('');
        return `<section class="lang-card${k === focusLang ? ' focus' : ''}" data-lang="${k}">
            <h2 style="background:${L.color}">${L.name}</h2>
            <div class="code-lines">${code}</div>
            <p class="says" aria-live="polite"></p>
            <p class="count"><span class="n">0</span> <span class="unit">${k === 'assembly' ? 'machine instructions' : 'steps'}</span> so far</p>
        </section>`;
    }).join('');
    phaseBar.innerHTML = PLAN.map((p, k) => `<span class="ph ph-${p.phase}" data-k="${k}" title="${PHASE_NAMES[p.phase]}${p.phase === 'end' ? '' : ' (i = ' + p.i + ')'}">${p.phase === 'init' ? 'set up' : p.phase === 'end' ? 'end' : p.phase}${p.phase === 'test' ? ' ' + p.i : ''}</span>`).join('');
}

function render() {
    const cur = PLAN[pos];
    document.querySelectorAll('.lang-card').forEach(card => {
        const k = card.dataset.lang, L = LANGS[k];
        card.querySelectorAll('.seg').forEach(seg => {
            const tags = seg.dataset.tags.split(' ');
            seg.classList.toggle('on', !!cur && tags.includes(cur.phase));
        });
        card.querySelector('.says').textContent = cur ? (typeof L.says[cur.phase] === 'function' ? L.says[cur.phase](cur.i) : L.says[cur.phase]) : '';
        let n = 0;
        for (let q = 0; q <= pos; q++) n += L.cost[PLAN[q].phase];
        card.querySelector('.n').textContent = n;
    });
    phaseBar.querySelectorAll('.ph').forEach(el => {
        const k = +el.dataset.k;
        el.classList.toggle('done', k < pos);
        el.classList.toggle('now', k === pos);
    });
    narration.innerHTML = cur
        ? `<b>${PHASE_NAMES[cur.phase]}</b>` + (cur.phase === 'end' ? ': the counter has reached 3.' : ` &middot; i = ${cur.i}`)
        : 'Press <b>Step</b> (or Space) to run the loop one phase at a time in all four languages.';

    // output and iteration circles
    const printed = PLAN.slice(0, pos + 1).filter(p => p.phase === 'body').map(p => `Iteration ${p.i}`);
    outputContainer.classList.toggle('hidden', printed.length === 0);
    if (outputLog.children.length !== printed.length) {   // only the newest line fades in
        const grew = printed.length === outputLog.children.length + 1;
        outputLog.innerHTML = printed.map((t, k) => `<div class="output-line${grew && k === printed.length - 1 ? ' new' : ''}">${t}</div>`).join('');
    }
    iterationContainer.classList.toggle('hidden', pos < 0);
    const iNow = cur ? cur.i : -1;
    circles.forEach(c => {
        const v = +c.dataset.value;
        c.classList.toggle('active', v === iNow && cur && cur.phase !== 'end');
        c.classList.toggle('past', v < iNow || (cur && cur.phase === 'end'));
    });
    btnStep.disabled = !!timer;
    btnExecute.innerHTML = timer ? '<i data-lucide="pause"></i> Pause' : '<i data-lucide="play"></i> Execute Loop';
    if (window.lucide) lucide.createIcons();
}

function step() {
    if (pos >= PLAN.length - 1) { pos = -1; }   // after the end, the next Step starts again
    else pos++;
    render();
}

function execute() {
    if (timer) { clearInterval(timer); timer = null; render(); return; }
    if (pos >= PLAN.length - 1) pos = -1;
    timer = setInterval(() => {
        pos++;
        if (pos >= PLAN.length - 1) { pos = PLAN.length - 1; clearInterval(timer); timer = null; }
        render();
    }, 900);
    pos++; render();
}

function reset() {
    if (timer) { clearInterval(timer); timer = null; }
    pos = -1; render();
}

btnStep.addEventListener('click', step);
btnExecute.addEventListener('click', execute);
btnReset.addEventListener('click', reset);
build();
render();
// ?steps=N is handled by course.js (it presses Step N times)
