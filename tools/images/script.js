// How a computer sees an image: pixels, three colour planes, and one flattened row of features.
const imageLoader = document.getElementById('imageLoader');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const pixelGrid = document.getElementById('pixel-grid');
const pixelZoom = document.getElementById('pixel-zoom');
const pixelWhere = document.getElementById('pixel-where');
const planesBox = document.getElementById('planes');
const flatTable = document.getElementById('flat');
const msg = document.getElementById('upload-msg');
const allowedImageTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp'];
const SIZES = [8, 16, 32];

const q = new URLSearchParams(location.search);
let gridSize = SIZES.includes(parseInt(q.get('grid'), 10)) ? parseInt(q.get('grid'), 10) : 8;
let source = null;          // the loaded picture (sample or upload), redrawn at every resolution
let randomPixels = null;    // or a random image, made once at 32 x 32 and shrunk from there
let selected = null;        // [row, col] of the selected pixel

// ?seed=42 makes the random image repeatable (small seeded generator, mulberry32)
let rand = Math.random;
(function () {
    const seed = parseInt(q.get('seed'), 10);
    if (Number.isNaN(seed)) return;
    let a = seed >>> 0;
    rand = function () {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
})();

// ---------- drawing the image at the chosen resolution
function draw() {
    canvas.width = gridSize;
    canvas.height = gridSize;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    if (source) {
        // centre crop to a square, then shrink to gridSize x gridSize
        const s = Math.min(source.width, source.height);
        ctx.drawImage(source, (source.width - s) / 2, (source.height - s) / 2, s, s, 0, 0, gridSize, gridSize);
    } else if (randomPixels) {
        // average blocks of the 32 x 32 random image
        const f = 32 / gridSize, img = ctx.createImageData(gridSize, gridSize);
        for (let y = 0; y < gridSize; y++) for (let x = 0; x < gridSize; x++) {
            const sum = [0, 0, 0];
            for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) {
                const k = ((y * f + dy) * 32 + (x * f + dx)) * 3;
                sum[0] += randomPixels[k]; sum[1] += randomPixels[k + 1]; sum[2] += randomPixels[k + 2];
            }
            const o = (y * gridSize + x) * 4;
            img.data[o] = Math.round(sum[0] / (f * f)); img.data[o + 1] = Math.round(sum[1] / (f * f)); img.data[o + 2] = Math.round(sum[2] / (f * f)); img.data[o + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }
    if (selected && (selected[0] >= gridSize || selected[1] >= gridSize)) selected = null;
    showAll();
}

function loadPicture(src) {
    const img = new Image();
    img.onload = () => { source = img; randomPixels = null; draw(); };
    img.src = src;
}
function makeRandom() {
    randomPixels = new Uint8ClampedArray(32 * 32 * 3).map(() => Math.floor(rand() * 256));
    source = null;
    draw();
}
function handleImage(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!allowedImageTypes.includes(file.type)) { msg.textContent = 'Please choose a PNG, JPG, GIF or BMP image.'; return; }
    msg.textContent = '';
    const reader = new FileReader();
    reader.onload = (event) => loadPicture(event.target.result);
    reader.readAsDataURL(file);
}

// ---------- the three views
const pix = () => ctx.getImageData(0, 0, gridSize, gridSize).data;

function showAll() {
    const d = pix(), n = gridSize;
    document.querySelectorAll('[data-grid]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.grid === n)));
    document.getElementById('size-note').textContent = `${n} × ${n} = ${n * n} pixels`;
    document.getElementById('shape').textContent = `(${n}, ${n}, 3)`;

    // 1. pixel grid
    pixelGrid.innerHTML = '';
    pixelGrid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    pixelGrid.style.gap = n > 16 ? '0' : n > 8 ? '1px' : '2px';
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const o = (y * n + x) * 4;
        const cell = document.createElement('div');
        cell.className = 'pixel';
        cell.style.backgroundColor = `rgb(${d[o]}, ${d[o + 1]}, ${d[o + 2]})`;
        cell.dataset.y = y; cell.dataset.x = x;
        cell.title = `row ${y + 1}, column ${x + 1}: R ${d[o]}, G ${d[o + 1]}, B ${d[o + 2]}`;
        pixelGrid.appendChild(cell);
    }

    // 2. three planes as number grids, shaded in their colour
    const names = [['Red', 'red', '188, 0, 49'], ['Green', 'green', '46, 125, 50'], ['Blue', 'blue', '63, 99, 145']];
    planesBox.innerHTML = names.map(([name, cls, rgb], c) => {
        let t = `<div class="plane"><h3 class="${cls}">${name}: ${n} × ${n} numbers</h3><div class="plane-scroll"><table class="plane-t size-${n}">`;
        for (let y = 0; y < n; y++) {
            t += '<tr>';
            for (let x = 0; x < n; x++) {
                const v = d[(y * n + x) * 4 + c];
                t += `<td data-y="${y}" data-x="${x}" style="background:rgba(${rgb},${(v / 255).toFixed(2)});color:${v > 140 ? '#FFFFFF' : '#1F1D21'}">${v}</td>`;
            }
            t += '</tr>';
        }
        return t + '</table></div></div>';
    }).join('');

    // 3. flattened row
    const len = n * n * 3;
    document.getElementById('flat-note').innerHTML =
        `To learn from images, a simple model reads each picture as <b>one row of ${len.toLocaleString('en')} numbers</b>: ` +
        `pixel 1's red, green and blue, then pixel 2's, and so on, row by row. That row is like one row of the features table in ` +
        `the <a href="../../code-along/week3/">Week 3 code-along</a>, with ${len.toLocaleString('en')} columns instead of a handful. Where each pixel sat in the picture is no longer visible.`;
    let head = '<tr><th class="rowhead">feature</th>', row = '<tr><td class="rowhead">value</td>';
    for (let k = 0; k < n * n; k++) {
        const o = k * 4;
        ['R', 'G', 'B'].forEach((ch, c) => {
            head += `<th class="${['red', 'green', 'blue'][c]}" data-k="${k}">p${k + 1} ${ch}</th>`;
            row += `<td data-k="${k}">${d[o + c]}</td>`;
        });
    }
    flatTable.innerHTML = head + '</tr>' + row + '</tr>';
    showSelected();
}

function showSelected() {
    document.querySelectorAll('.pixel.selected, .plane-t td.sel, .flat .sel').forEach(e => e.classList.remove('selected', 'sel'));
    if (!selected) {
        pixelZoom.style.backgroundColor = '';
        pixelZoom.innerHTML = '<span class="hint">Click a pixel</span>';
        pixelWhere.textContent = '';
        return;
    }
    const [y, x] = selected, n = gridSize, k = y * n + x, d = pix(), o = k * 4;
    const cell = pixelGrid.children[k];
    if (cell) cell.classList.add('selected');
    document.querySelectorAll(`.plane-t td[data-y="${y}"][data-x="${x}"]`).forEach(td => td.classList.add('sel'));
    document.querySelectorAll(`.flat [data-k="${k}"]`).forEach(e => e.classList.add('sel'));
    pixelZoom.style.backgroundColor = `rgb(${d[o]}, ${d[o + 1]}, ${d[o + 2]})`;
    pixelZoom.innerHTML = `<div class="rgb-value"><span class="red">R:</span> ${d[o]}</div><div class="rgb-value"><span class="green">G:</span> ${d[o + 1]}</div><div class="rgb-value"><span class="blue">B:</span> ${d[o + 2]}</div>`;
    pixelWhere.innerHTML = `Row ${y + 1}, column ${x + 1}: pixel ${k + 1} of ${n * n}.<br>In the flattened row: columns ${3 * k + 1} to ${3 * k + 3}.`;
    const th = flatTable.querySelector(`th[data-k="${k}"]`);
    if (th) {   // scroll the row sideways only, never the page
        const wrap = flatTable.parentElement;
        wrap.scrollLeft = th.offsetLeft - wrap.clientWidth / 2 + th.offsetWidth * 1.5;
    }
}

pixelGrid.addEventListener('click', e => {
    const c = e.target.closest('.pixel'); if (!c) return;
    selected = [+c.dataset.y, +c.dataset.x]; showSelected();
});
planesBox.addEventListener('click', e => {
    const td = e.target.closest('td[data-y]'); if (!td) return;
    selected = [+td.dataset.y, +td.dataset.x]; showSelected();
});
document.querySelectorAll('[data-grid]').forEach(b => b.addEventListener('click', () => {
    if (selected) selected = [Math.floor(selected[0] * +b.dataset.grid / gridSize), Math.floor(selected[1] * +b.dataset.grid / gridSize)];
    gridSize = +b.dataset.grid; draw();
}));
document.getElementById('gridNext').addEventListener('click', () => {
    const next = SIZES[(SIZES.indexOf(gridSize) + 1) % SIZES.length];
    if (selected) selected = [Math.floor(selected[0] * next / gridSize), Math.floor(selected[1] * next / gridSize)];
    gridSize = next; draw();
});
imageLoader.addEventListener('change', handleImage);
document.getElementById('uploadBtn').addEventListener('click', () => imageLoader.click());
document.getElementById('randomBtn').addEventListener('click', makeRandom);
document.getElementById('sampleBtn').addEventListener('click', () => loadPicture('sample-cat.jpg'));

// start: the sample photo, or a random image with ?source=random or ?seed=
if (q.get('source') === 'random' || q.has('seed')) makeRandom();
else loadPicture('sample-cat.jpg');
