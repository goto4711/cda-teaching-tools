// Term frequency, TF-IDF and document similarity for three short documents.
// The TF-IDF formula is the one from the Week 4 code-along: tf × ln(N / df).
document.addEventListener('DOMContentLoaded', () => {
    const docs = [1, 2, 3].map(i => document.getElementById('doc' + i));
    const stopwordsInput = document.getElementById('stopwords');
    const output = document.getElementById('output');
    const modeButtons = document.querySelectorAll('[data-mode]');
    const exampleSelect = document.getElementById('example');

    const EXAMPLES = {
        fox: {
            label: 'The quick brown fox',
            docs: ['The quick brown fox jumps over the lazy dog.', 'Never jump over the lazy dog.', 'A quick brown dog jumps over the lazy fox.'],
            stop: 'the,a,is,in,it,of,for,with,over'
        },
        bites: {
            label: 'Who bites whom?',
            docs: ['Dog bites man.', 'Man bites dog.', 'The man was bitten by the dog.'],
            stop: 'the,a,is,in,it,of,for,with,over'
        },
        house: {
            label: 'House and garden (Week 4 code-along)',
            docs: ['The house has a beautiful garden with many flowers.', 'A small house has a red roof and a big backyard.', 'The garden has green grass and colorful plants around the house.'],
            stop: 'the,a,has,and,with,many,around'
        }
    };
    Object.entries(EXAMPLES).forEach(([k, e]) => {
        const o = document.createElement('option'); o.value = k; o.textContent = e.label; exampleSelect.appendChild(o);
    });
    const custom = document.createElement('option'); custom.value = 'custom'; custom.textContent = 'Your own text'; exampleSelect.appendChild(custom);

    // ?example=bites, ?d1=..&d2=..&d3=..&stop=.., ?mode=tfidf
    const q = new URLSearchParams(location.search);
    let mode = q.get('mode') === 'tfidf' ? 'tfidf' : 'tf';
    function loadExample(k) {
        const e = EXAMPLES[k]; if (!e) return;
        docs.forEach((d, i) => { d.value = e.docs[i]; });
        stopwordsInput.value = e.stop;
        exampleSelect.value = k;
    }
    loadExample(EXAMPLES[q.get('example')] ? q.get('example') : 'fox');
    let edited = false;
    [['d1', docs[0]], ['d2', docs[1]], ['d3', docs[2]], ['stop', stopwordsInput]].forEach(([k, box]) => {
        if (q.has(k)) { box.value = q.get(k); edited = true; }
    });
    if (edited) exampleSelect.value = 'custom';

    const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const tokens = text => text.toLowerCase().match(/\b(\w+)\b/g) || [];
    const fmt = v => (Number.isInteger(v) ? String(v) : v.toFixed(2));

    function cosine(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return na && nb ? dot / Math.sqrt(na * nb) : 0;
    }

    function render() {
        const stop = new Set(stopwordsInput.value.toLowerCase().split(',').map(w => w.trim()).filter(Boolean));
        const all = docs.map(d => tokens(d.value));
        const kept = all.map(ws => ws.filter(w => !stop.has(w)));
        const vocab = [...new Set(kept.flat())].sort();
        const N = docs.length;
        const tf = kept.map(ws => vocab.map(v => ws.filter(w => w === v).length));
        const df = vocab.map((_, j) => tf.filter(row => row[j] > 0).length);
        const idf = df.map(d => Math.log(N / d));
        const values = mode === 'tfidf' ? tf.map(row => row.map((c, j) => c * idf[j])) : tf;

        let html = '';
        // 1. what the computer reads
        html += '<h3>1. What the computer keeps</h3><div class="reads">';
        all.forEach((ws, i) => {
            html += `<div class="read-row"><span class="read-doc">Doc ${i + 1}</span><span>` +
                (ws.length ? ws.map(w => stop.has(w) ? `<s class="stop" title="stopword: removed">${esc(w)}</s>` : `<span class="kept">${esc(w)}</span>`).join(' ') : '<i class="muted">empty</i>') +
                '</span></div>';
        });
        const removed = all.flat().filter(w => stop.has(w)).length;
        html += `<p class="note">Lower case, punctuation gone, split into words. ${removed} stopword${removed === 1 ? '' : 's'} struck through and removed.</p></div>`;

        // 2. the table
        html += `<h3>2. ${mode === 'tfidf' ? 'TF-IDF weights' : 'Term frequency: word counts'}</h3>`;
        html += '<div class="table-container"><table class="tf"><tr><th>Document</th>' + vocab.map(v => `<th>${esc(v)}</th>`).join('') + '</tr>';
        values.forEach((row, i) => {
            html += `<tr><td class="rowhead">Doc ${i + 1}</td>` + row.map(v => `<td class="${v > 0 ? 'positive' : 'zero'}">${fmt(v)}</td>`).join('') + '</tr>';
        });
        if (mode === 'tfidf') {
            html += '<tr class="aux"><td class="rowhead">in how many docs (df)</td>' + df.map(d => `<td>${d}</td>`).join('') + '</tr>';
            html += `<tr class="aux"><td class="rowhead">idf = ln(${N} / df)</td>` + idf.map(v => `<td>${v.toFixed(2)}</td>`).join('') + '</tr>';
        }
        html += '</table></div>';
        if (mode === 'tfidf') {
            const everywhere = vocab.filter((_, j) => df[j] === N);
            html += '<p class="note">TF-IDF = count &times; ln(3 / number of documents containing the word). ' +
                (everywhere.length ? `Words in all three documents get 0, however often they occur: <b>${everywhere.map(esc).join(', ')}</b>.` : 'No word occurs in all three documents.') + '</p>';
        } else {
            html += `<p class="note">${vocab.length} words in the vocabulary, so each document becomes a row of ${vocab.length} numbers: its vector.</p>`;
        }

        // 3. similarity
        html += '<h3>3. How similar are the documents?</h3>';
        html += '<div class="table-container"><table class="sim"><tr><th></th>' + [1, 2, 3].map(i => `<th>Doc ${i}</th>`).join('') + '</tr>';
        const sims = [];
        values.forEach((a, i) => {
            html += `<tr><td class="rowhead">Doc ${i + 1}</td>`;
            values.forEach((b, j) => {
                const s = cosine(a, b);
                if (j > i) sims.push({ i, j, s });
                const shade = i === j ? '#F4F4F4' : `rgba(188, 0, 49, ${(s * 0.85).toFixed(2)})`;
                const ink = i !== j && s > 0.55 ? '#FFFFFF' : '#1F1D21';
                html += `<td style="background:${shade};color:${ink}" class="${i === j ? 'diag' : ''}">${s.toFixed(2)}</td>`;
            });
            html += '</tr>';
        });
        html += '</table></div>';
        sims.sort((a, b) => b.s - a.s);
        const same = sims.filter(p => p.s > 0.9999);
        let msg = 'Cosine similarity: 1 = the same mix of words, 0 = no words in common. ';
        if (same.length) msg += same.map(p => `<b>Doc ${p.i + 1} and Doc ${p.j + 1} come out identical</b>` + (docs[p.i].value.trim() === docs[p.j].value.trim() ? ' (the texts are the same).' : ', although the texts differ.')).join(' ');
        else if (sims.length) msg += `Most similar: Doc ${sims[0].i + 1} and Doc ${sims[0].j + 1} (${sims[0].s.toFixed(2)}). Least similar: Doc ${sims[sims.length - 1].i + 1} and Doc ${sims[sims.length - 1].j + 1} (${sims[sims.length - 1].s.toFixed(2)}).`;
        html += `<p class="note" data-k="sim-note">${msg}</p>`;
        output.innerHTML = html;
        modeButtons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    }

    let t = null;
    const later = () => { clearTimeout(t); t = setTimeout(render, 200); };
    docs.concat([stopwordsInput]).forEach(box => box.addEventListener('input', () => { exampleSelect.value = 'custom'; later(); }));
    exampleSelect.addEventListener('change', () => { loadExample(exampleSelect.value); render(); });
    modeButtons.forEach(b => b.addEventListener('click', () => { mode = b.dataset.mode; render(); }));
    document.getElementById('processBtn').addEventListener('click', render);
    render();
});
