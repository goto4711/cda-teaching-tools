/**
 * App Controller
 */

const memorySize = 100;
const memory = new Memory(memorySize);
const cpu = new CPU(memory);

// DOM Elements
const els = {
    memoryGrid: document.getElementById('memory-display'),
    pc: document.querySelector('#comp-pc .value'),
    mar: document.querySelector('#comp-mar .value'),
    ir: document.querySelector('#comp-ir .value'),
    irDecode: document.getElementById('ir-decode'),
    mdr: document.querySelector('#comp-mdr .value'),
    acc: document.querySelector('#comp-acc .value'),
    aluOp: document.getElementById('alu-op'),
    cuState: document.getElementById('cu-state'),
    outputLog: document.getElementById('output-log'),
    executionLog: document.getElementById('execution-log'),
    inputBuffer: document.getElementById('input-buffer'),
    btnSubmitInput: document.getElementById('btn-submit-input'),
    btnStep: document.getElementById('btn-step'),
    btnRun: document.getElementById('btn-run'),
    btnReset: document.getElementById('btn-reset'),
    btnLoad: document.getElementById('btn-load'),
    programSelect: document.getElementById('program-select'),
    speedRange: document.getElementById('speed-range'),
    btnOpcodes: document.getElementById('btn-opcodes'),
    modal: document.getElementById('opcode-modal'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    opcodeListBody: document.getElementById('opcode-list-body'),
    buses: {
        data: document.getElementById('data-bus'),
        addr: document.getElementById('address-bus'),
        control: document.getElementById('control-bus')
    },
    components: {
        pc: document.getElementById('comp-pc'),
        mar: document.getElementById('comp-mar'),
        ir: document.getElementById('comp-ir'),
        mdr: document.getElementById('comp-mdr'),
        acc: document.getElementById('comp-acc'),
        alu: document.getElementById('comp-alu'),
        cu: document.getElementById('comp-cu')
    }
};

let isRunning = false;
let runInterval = null;
let speed = 1000;

// Helper: Convert to Binary String
function toBinary(num, bits) {
    return (num >>> 0).toString(2).padStart(bits, '0').slice(-bits);
}

// Initialize Memory Grid
function initMemoryGrid() {
    els.memoryGrid.innerHTML = '';
    for (let i = 0; i < memorySize; i++) {
        const cell = document.createElement('div');
        cell.className = 'memory-cell';
        cell.id = `mem-${i}`;
        cell.innerHTML = `
            <div class="cell-addr">${toBinary(i, 8)} <span class="cell-no">#${i}</span></div>
            <div class="cell-val">0000 000000000000</div>
            <div class="cell-meaning"></div>
        `;
        els.memoryGrid.appendChild(cell);
    }
}

// Update UI from CPU State
function updateUI() {
    // Update Components
    els.pc.textContent = toBinary(cpu.pc, 8);
    els.mar.textContent = toBinary(cpu.mar, 8);
    els.ir.textContent = toBinary(cpu.ir, 16).replace(/^(.{4})/, '$1 ');
    els.mdr.textContent = toBinary(cpu.mdr, 16).replace(/^(.{4})/, '$1 ');
    els.acc.textContent = toBinary(cpu.acc, 16);

    // Decode IR hint
    const opcode = (cpu.ir >> 12) & 0xF;
    const instr = OpcodeMap[opcode];
    els.irDecode.textContent = describeWord(cpu.ir);

    // ALU State
    els.aluOp.textContent = cpu.state === 'EXECUTE' && instr ? instr.name : 'Idle';

    // CU State
    els.cuState.textContent = cpu.state;

    // Buses used in the last step
    Object.entries(els.buses).forEach(([k, el]) => { if (el) el.classList.toggle('active', cpu.activeBuses.includes(k)); });

    // Memory Grid Highlights
    document.querySelectorAll('.memory-cell').forEach(c => {
        c.classList.remove('active-read', 'active-write', 'pc-highlight');
    });

    // Highlight PC position
    const pcCell = document.getElementById(`mem-${cpu.pc}`);
    if (pcCell) pcCell.classList.add('pc-highlight');

    // Highlight Memory Access
    if (memory.lastAccess !== -1) {
        const cell = document.getElementById(`mem-${memory.lastAccess}`);
        if (cell) {
            cell.classList.add(memory.lastOperation === 'read' ? 'active-read' : 'active-write');
        }
    }

    // Update Memory Values: the bits, split into opcode | operand, and what they mean
    memory.data.forEach((val, i) => {
        const cell = document.getElementById(`mem-${i}`);
        if (!cell) return;
        const bits = toBinary(val, 16);
        cell.querySelector('.cell-val').textContent = bits.slice(0, 4) + ' ' + bits.slice(4);
        const meaning = cell.querySelector('.cell-meaning');
        const asInstr = describeWord(val);
        const asNumber = String(val);
        const isCode = i < programLength;
        meaning.textContent = val === 0 && !isCode ? '' : isCode ? '→ ' + asInstr : '= ' + asNumber;
        meaning.className = 'cell-meaning ' + (isCode ? 'is-code' : 'is-data');
        cell.title = `Address ${i}. As an instruction: ${asInstr}. As a number: ${asNumber}. The same bits; the PC decides which.`;
    });

    // Output Log
    els.outputLog.innerHTML = cpu.outputBuffer.map(v => `> ${toBinary(v, 16)} (${v})`).join('<br>');
    els.outputLog.scrollTop = els.outputLog.scrollHeight;

    // Input State
    if (cpu.waitingForInput) {
        els.inputBuffer.disabled = false;
        els.btnSubmitInput.disabled = false;
        els.inputBuffer.focus();
        log("Waiting for input...");
    } else {
        els.inputBuffer.disabled = true;
        els.btnSubmitInput.disabled = true;
    }

    // Halted State
    if (cpu.halted) {
        stopRun();
        log("CPU Halted.");
    }
}

function log(msg) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.textContent = msg;
    els.executionLog.appendChild(div);
    els.executionLog.scrollTop = els.executionLog.scrollHeight;
}

// The same 16 bits read as an instruction: 4-bit opcode + 12-bit operand
function describeWord(val) {
    const instr = OpcodeMap[(val >> 12) & 0xF];
    if (!instr) return '???';
    const operand = val & 0xFFF;
    return ['LOAD', 'ADD', 'SUB', 'STO', 'JMP', 'JZ'].includes(instr.name) ? `${instr.name} ${operand}` : instr.name;
}
let programLength = 0; // cells 0 .. programLength-1 hold the loaded program

// Programs
// Helper to encode instructions: Opcode (4 bits) | Operand (12 bits)
const _ = (op, val) => (op << 12) | val;

const Programs = {
    empty: [],
    add: [
        _(1, 90), // LOAD 90
        _(2, 91), // ADD 91
        _(4, 92), // STO 92
        _(15, 0)  // HALT
    ],
    loop: [
        _(1, 90), // 0 LOAD 90   start value
        _(4, 91), // 1 STO 91    counter = start
        _(1, 91), // 2 LOAD 91   ACC = counter
        _(6, 8),  // 3 JZ 08     counter is 0? jump to HALT (was JZ 06, which pointed at JMP and never ended)
        _(8, 0),  // 4 OUT       show the counter
        _(3, 92), // 5 SUB 92    ACC = counter - 1
        _(4, 91), // 6 STO 91    counter = ACC
        _(5, 2),  // 7 JMP 02    back to the test
        _(15, 0)  // 8 HALT
    ],
    input: [
        _(7, 0),  // IN
        _(8, 0),  // OUT
        _(5, 0)   // JMP 00
    ]
};

// Data initialization for programs
function loadProgramData(name) {
    if (name === 'add') {
        memory.write(90, 15);
        memory.write(91, 27);
    } else if (name === 'loop') {
        memory.write(90, 5); // Start count
        memory.write(92, 1); // Decrement value
    }
}

// Event Listeners
function stepOnce() {
    if (cpu.halted || cpu.waitingForInput) return;
    cpu.step();
    if (cpu.lastAction) log(cpu.lastAction);
    updateUI();
}
els.btnStep.addEventListener('click', stepOnce);

els.btnRun.addEventListener('click', () => {
    if (isRunning) {
        stopRun();
    } else {
        startRun();
    }
});

els.btnReset.addEventListener('click', () => {
    stopRun();
    cpu.reset();
    memory.reset();
    // Reload current program
    const progName = els.programSelect.value;
    if (Programs[progName]) {
        memory.loadProgram(Programs[progName]);
        loadProgramData(progName);
        programLength = Programs[progName].length;
    }
    els.executionLog.innerHTML = '';
    log("System Reset.");
    updateUI();
});

els.btnLoad.addEventListener('click', () => {
    stopRun();
    cpu.reset();
    memory.reset();
    const progName = els.programSelect.value;
    if (Programs[progName]) {
        memory.loadProgram(Programs[progName]);
        loadProgramData(progName);
        programLength = Programs[progName].length;
        log(`Loaded program: ${progName}`);
    }
    updateUI();
});

els.btnSubmitInput.addEventListener('click', () => {
    const val = els.inputBuffer.value;
    if (val !== '') {
        cpu.provideInput(val);
        els.inputBuffer.value = '';
        log(`Input provided: ${val}`);
        updateUI();
        if (isRunning) startRun(); // Resume running if we were running
    }
});

els.speedRange.addEventListener('input', (e) => {
    speed = 2100 - e.target.value; // Invert so right is faster
    if (isRunning) {
        stopRun();
        startRun();
    }
});

function startRun() {
    if (cpu.halted) return;
    isRunning = true;
    els.btnRun.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15V9"/><path d="M14 15V9"/></svg> Pause`;
    els.btnRun.classList.remove('accent');
    els.btnRun.classList.add('secondary'); // Visual toggle

    runInterval = setInterval(() => {
        if (cpu.waitingForInput || cpu.halted) {
            stopRun();
            return;
        }
        stepOnce();
    }, speed);
}

function stopRun() {
    isRunning = false;
    clearInterval(runInterval);
    els.btnRun.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Run`;
    els.btnRun.classList.add('accent');
    els.btnRun.classList.remove('secondary');
}

// Opcode List Logic
function initOpcodeList() {
    els.opcodeListBody.innerHTML = '';
    Object.values(IS).forEach(instr => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${toBinary(instr.opcode, 4)}</td>
            <td>${instr.name}</td>
            <td>${instr.desc}</td>
        `;
        els.opcodeListBody.appendChild(row);
    });
}

els.btnOpcodes.addEventListener('click', () => {
    els.modal.classList.remove('hidden');
});

els.btnCloseModal.addEventListener('click', () => {
    els.modal.classList.add('hidden');
});

els.modal.addEventListener('click', (e) => {
    if (e.target === els.modal) {
        els.modal.classList.add('hidden');
    }
});

// Enter in the input box sends the value
els.inputBuffer.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); els.btnSubmitInput.click(); }
});

// Init
initMemoryGrid();
initOpcodeList();
updateUI();
log("System Ready.");

// Deep link: ?program=add|loop|input loads that program (course.js then applies ?steps=N)
(function () {
    const prog = new URLSearchParams(location.search).get('program');
    if (prog && Programs[prog] && prog !== 'empty') {
        els.programSelect.value = prog;
        els.btnLoad.click();
    }
})();
