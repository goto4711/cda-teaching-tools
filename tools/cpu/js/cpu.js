/**
 * Von Neumann CPU Simulation Logic
 */

// Instruction Set Architecture
const IS = {
    NOP: { opcode: 0, name: 'NOP', desc: 'No Operation' },
    LOAD: { opcode: 1, name: 'LOAD', desc: 'Load memory to ACC' },
    ADD: { opcode: 2, name: 'ADD', desc: 'Add memory to ACC' },
    SUB: { opcode: 3, name: 'SUB', desc: 'Subtract memory from ACC' },
    STO: { opcode: 4, name: 'STO', desc: 'Store ACC to memory' },
    JMP: { opcode: 5, name: 'JMP', desc: 'Jump to address' },
    JZ: { opcode: 6, name: 'JZ', desc: 'Jump if ACC is zero' },
    IN: { opcode: 7, name: 'IN', desc: 'Input to ACC' },
    OUT: { opcode: 8, name: 'OUT', desc: 'Output from ACC' },
    HALT: { opcode: 15, name: 'HALT', desc: 'Stop execution' }
};

// Reverse lookup for decoding
const OpcodeMap = {};
Object.values(IS).forEach(instr => OpcodeMap[instr.opcode] = instr);

class Memory {
    constructor(size = 100) {
        this.size = size;
        this.data = new Array(size).fill(0);
        this.lastAccess = -1; // For visualization
        this.lastOperation = null; // 'read' or 'write'
    }

    reset() {
        this.data.fill(0);
        this.lastAccess = -1;
        this.lastOperation = null;
    }

    read(address) {
        if (address < 0 || address >= this.size) return 0;
        this.lastAccess = address;
        this.lastOperation = 'read';
        return this.data[address];
    }

    write(address, value) {
        if (address < 0 || address >= this.size) return;
        this.lastAccess = address;
        this.lastOperation = 'write';
        this.data[address] = value & 0xFFFF; // 16-bit limit
    }

    loadProgram(program) {
        this.reset();
        program.forEach((val, index) => {
            if (index < this.size) this.data[index] = val;
        });
    }
}

class CPU {
    constructor(memory) {
        this.memory = memory;
        this.reset();
    }

    reset() {
        this.pc = 0;    // Program Counter
        this.acc = 0;   // Accumulator
        this.ir = 0;    // Instruction Register
        this.mar = 0;   // Memory Address Register
        this.mdr = 0;   // Memory Data Register
        this.state = 'FETCH'; // FETCH, DECODE, EXECUTE, HALTED, WAIT_INPUT
        this.halted = false;
        this.waitingForInput = false;
        this.outputBuffer = []; // Store output history

        // For visualization: track active components/buses
        this.activeBus = null; // 'addr', 'data', 'control'
        this.activeBuses = []; // all buses used in the last step
        this.activeComponent = null;
        this.lastAction = '';  // one-line explanation of the last step, shown in the log
    }

    step() {
        if (this.halted || this.waitingForInput) return false;

        switch (this.state) {
            case 'FETCH':
                this.fetch();
                break;
            case 'DECODE':
                this.decode();
                break;
            case 'EXECUTE':
                this.execute();
                break;
        }
        return true;
    }

    fetch() {
        // 1. PC -> MAR
        this.mar = this.pc;
        this.activeBus = 'addr';
        this.activeComponent = 'pc';

        // 2. Memory Read -> MDR
        this.mdr = this.memory.read(this.mar);

        // 3. MDR -> IR
        this.ir = this.mdr;

        // 4. Increment PC
        this.pc++;

        this.activeBuses = ['addr', 'data'];
        this.lastAction = `FETCH: PC ${this.mar} → MAR; memory[${this.mar}] → MDR → IR; PC becomes ${this.pc}`;
        this.state = 'DECODE';
    }

    decode() {
        // Decode opcode and operand using bitwise operations
        // Format: OOOO AAAAAAAAAAAA (4 bits Opcode, 12 bits Address/Operand)

        this.opcode = (this.ir >> 12) & 0xF;
        this.operand = this.ir & 0xFFF;

        this.activeComponent = 'ir';
        this.activeBuses = ['control'];
        const d = OpcodeMap[this.opcode];
        this.lastAction = `DECODE: opcode ${this.opcode.toString(2).padStart(4, '0')} = ${d ? d.name : '???'}, operand ${this.operand}`;
        this.state = 'EXECUTE';
    }

    execute() {
        const instr = OpcodeMap[this.opcode];
        this.activeComponent = 'alu'; // Default, might change
        this.activeBuses = ['control'];
        const usesMemory = instr && ['LOAD', 'ADD', 'SUB', 'STO'].includes(instr.name);
        if (usesMemory) this.activeBuses = ['addr', 'data', 'control'];
        this.lastAction = instr ? this.describe(instr) : 'EXECUTE: unknown opcode, CPU halts';

        if (!instr) {
            // Invalid opcode, treat as NOP or HALT? Let's HALT.
            this.halted = true;
            this.state = 'HALTED';
            return;
        }

        switch (instr.name) {
            case 'NOP':
                break;
            case 'LOAD':
                this.mar = this.operand;
                this.acc = this.memory.read(this.mar);
                break;
            case 'ADD':
                this.mar = this.operand;
                this.acc += this.memory.read(this.mar);
                break;
            case 'SUB':
                this.mar = this.operand;
                this.acc -= this.memory.read(this.mar);
                break;
            case 'STO':
                this.mar = this.operand;
                this.memory.write(this.mar, this.acc);
                break;
            case 'JMP':
                this.pc = this.operand;
                break;
            case 'JZ':
                if (this.acc === 0) {
                    this.pc = this.operand;
                }
                break;
            case 'IN':
                this.waitingForInput = true;
                this.state = 'WAIT_INPUT'; // Pause execution
                return; // Don't finish cycle yet
            case 'OUT':
                this.outputBuffer.push(this.acc);
                break;
            case 'HALT':
                this.halted = true;
                this.state = 'HALTED';
                return;
        }

        this.state = 'FETCH';
    }

    describe(instr) {
        const a = this.operand, m = this.memory.data[a];
        switch (instr.name) {
            case 'NOP': return 'EXECUTE NOP: do nothing';
            case 'LOAD': return `EXECUTE LOAD ${a}: ACC = memory[${a}] = ${m}`;
            case 'ADD': return `EXECUTE ADD ${a}: ACC = ${this.acc} + memory[${a}] (${m}) = ${this.acc + m}`;
            case 'SUB': return `EXECUTE SUB ${a}: ACC = ${this.acc} − memory[${a}] (${m}) = ${this.acc - m}`;
            case 'STO': return `EXECUTE STO ${a}: memory[${a}] = ACC = ${this.acc}`;
            case 'JMP': return `EXECUTE JMP ${a}: PC = ${a}`;
            case 'JZ': return this.acc === 0 ? `EXECUTE JZ ${a}: ACC is 0, so PC = ${a}` : `EXECUTE JZ ${a}: ACC is ${this.acc}, not 0, so carry on`;
            case 'IN': return 'EXECUTE IN: wait for a value in the Input box';
            case 'OUT': return `EXECUTE OUT: send ACC (${this.acc}) to Output`;
            case 'HALT': return 'EXECUTE HALT: stop';
        }
        return '';
    }

    provideInput(value) {
        if (this.waitingForInput) {
            this.acc = parseInt(value) || 0;
            this.waitingForInput = false;
            this.state = 'FETCH'; // Resume
        }
    }
}
