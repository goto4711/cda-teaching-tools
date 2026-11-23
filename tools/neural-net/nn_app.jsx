
        const { useState, useEffect, useRef } = React;

        // ---- data: four made-up features per animal, label 0 = cat, 1 = dog
        const FEATURES = ["Ear Pointiness", "Whisker Length", "Nose Wetness", "Fur Pattern"];
        const CLEAR = [
          { features: [0.9, 0.8, 0.3, 0.7], label: 0, name: "Cat 1" },
          { features: [0.85, 0.9, 0.2, 0.8], label: 0, name: "Cat 2" },
          { features: [0.95, 0.75, 0.25, 0.75], label: 0, name: "Cat 3" },
          { features: [0.8, 0.85, 0.3, 0.65], label: 0, name: "Cat 4" },
          { features: [0.2, 0.3, 0.8, 0.4], label: 1, name: "Dog 1" },
          { features: [0.3, 0.25, 0.75, 0.35], label: 1, name: "Dog 2" },
          { features: [0.15, 0.35, 0.85, 0.3], label: 1, name: "Dog 3" },
          { features: [0.25, 0.2, 0.9, 0.45], label: 1, name: "Dog 4" },
        ];
        // the overlapping set adds animals that break the easy rule "pointy ears = cat"
        const OVERLAP = CLEAR.concat([
          { features: [0.25, 0.8, 0.35, 0.7], label: 0, name: "Cat 5", note: "folded ears" },
          { features: [0.9, 0.35, 0.8, 0.45], label: 1, name: "Dog 5", note: "pointy ears" },
          { features: [0.55, 0.5, 0.55, 0.5], label: 0, name: "Cat 6", note: "same numbers as Dog 6" },
          { features: [0.55, 0.5, 0.55, 0.5], label: 1, name: "Dog 6", note: "same numbers as Cat 6" },
        ]);
        // never trained on: the network only sees these to be tested
        const TEST = [
          { features: [0.8, 0.7, 0.35, 0.6], label: 0, name: "Cat A" },
          { features: [0.3, 0.3, 0.7, 0.4], label: 1, name: "Dog A" },
          { features: [0.3, 0.75, 0.3, 0.75], label: 0, name: "Cat B", note: "folded ears" },
          { features: [0.8, 0.3, 0.75, 0.4], label: 1, name: "Dog B", note: "pointy ears" },
        ];
        const DATASETS = {
          overlap: { data: OVERLAP, label: "Overlapping (12 animals)" },
          clear: { data: CLEAR, label: "Clear-cut (8 animals)" },
        };

        // small seeded random generator (mulberry32), so ?seed=7 gives the same start every time
        const makeRng = (seed) => {
          let a = seed >>> 0;
          return () => {
            a = (a + 0x6D2B79F5) >>> 0;
            let t = a;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
          };
        };
        const params = new URLSearchParams(location.search);
        const seedParam = parseInt(params.get("seed"), 10);
        const startData = params.get("data") === "clear" ? "clear" : "overlap";

        const relu = (x) => Math.max(0, x);
        const reluDeriv = (x) => (x > 0 ? 1 : 0);
        const softmax = (arr) => {
          const max = Math.max(...arr);
          const exps = arr.map((x) => Math.exp(x - max));
          const sum = exps.reduce((a, b) => a + b, 0);
          return exps.map((x) => x / sum);
        };
        const forward = (input, w1, b1, w2, b2) => {
          const hidden = b1.map((bias, i) => relu(input.reduce((acc, val, j) => acc + val * w1[j][i], 0) + bias));
          const output = b2.map((bias, i) => hidden.reduce((acc, val, j) => acc + val * w2[j][i], 0) + bias);
          return { hidden, output, probs: softmax(output) };
        };
        const evaluate = (set, w1, b1, w2, b2) => {
          let correct = 0, loss = 0;
          set.forEach((s) => {
            const p = forward(s.features, w1, b1, w2, b2).probs;
            if ((p[0] > p[1] ? 0 : 1) === s.label) correct++;
            loss += -Math.log(p[s.label] + 1e-10);
          });
          return { acc: (correct / set.length) * 100, loss: loss / set.length };
        };

        // ---- one animal, step by step: the same maths as one update in trainStep, kept for display
        // "push" = how much a node should change for the right answer (minus the gradient)
        const walkThrough = (sample, w1, b1, w2, b2, lr) => {
          const x = sample.features;
          const hiddenRaw = b1.map((bias, i) => x.reduce((acc, v, k) => acc + v * w1[k][i], 0) + bias);
          const hidden = hiddenRaw.map(relu);
          const outputRaw = b2.map((bias, j) => hidden.reduce((acc, v, i) => acc + v * w2[i][j], 0) + bias);
          const probs = softmax(outputRaw);
          const target = [0, 1].map((j) => (j === sample.label ? 1 : 0));
          const outPush = probs.map((p, j) => target[j] - p);
          const parts = hidden.map((_, i) => outPush.map((op, j) => w2[i][j] * op));
          const hidPush = hidden.map((_, i) => (hiddenRaw[i] > 0 ? parts[i][0] + parts[i][1] : 0));
          const nw1 = w1.map((row, k) => row.map((w, i) => w + lr * x[k] * hidPush[i]));
          const nw2 = w2.map((row, i) => row.map((w, j) => w + lr * hidden[i] * outPush[j]));
          const nb1 = b1.map((b, i) => b + lr * hidPush[i]);
          const nb2 = b2.map((b, j) => b + lr * outPush[j]);
          const moved = [];
          for (let k = 0; k < 4; k++) for (let i = 0; i < 6; i++) moved.push({ layer: 1, i: k, j: i, delta: nw1[k][i] - w1[k][i], from: x[k], fromName: FEATURES[k], push: hidPush[i], toName: "H" + (i + 1), before: w1[k][i] });
          for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) moved.push({ layer: 2, i, j, delta: nw2[i][j] - w2[i][j], from: hidden[i], fromName: "H" + (i + 1), push: outPush[j], toName: j === 0 ? "Cat" : "Dog", before: w2[i][j] });
          moved.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
          const after = forward(x, nw1, nb1, nw2, nb2);
          return { sample, x, hiddenRaw, hidden, probs, target, outPush, parts, hidPush, nw1, nw2, nb1, nb2, top: moved.slice(0, 5), after, lr, w2 };
        };
        const STAGES = [
          "Choose an animal",
          "Forward: the network guesses",
          "The error at the output",
          "The blame goes back",
          "Every weight takes a small step",
          "Check: guess again",
        ];
        const sgn = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

        const LossChart = ({ history, maxEpochs }) => {
          const W = 320, H = 170, L = 34, R = 8, T = 10, B = 26;
          const top = Math.max(1, ...history.map((h) => Math.max(h.loss, h.testLoss)));
          const x = (e) => L + (e / Math.max(maxEpochs, 1)) * (W - L - R);
          const y = (v) => T + (1 - Math.min(v, top) / top) * (H - T - B);
          const line = (key) => history.map((h, i) => (i ? "L" : "M") + x(h.epoch).toFixed(1) + " " + y(h[key]).toFixed(1)).join(" ");
          const coin = Math.log(2);
          return (
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Loss per epoch for training and unseen animals">
              <line x1={L} y1={T} x2={L} y2={H - B} stroke="#BEC6D1" />
              <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="#BEC6D1" />
              <line x1={L} y1={y(coin)} x2={W - R} y2={y(coin)} stroke="#8A8A8A" strokeDasharray="3 3" />
              <text x={W - R} y={y(coin) - 3} fontSize="9" textAnchor="end" fill="#555555">coin toss (0.69)</text>
              <text x={L - 4} y={T + 4} fontSize="9" textAnchor="end" fill="#555555">{top.toFixed(1)}</text>
              <text x={L - 4} y={H - B} fontSize="9" textAnchor="end" fill="#555555">0</text>
              <text x={L} y={H - 8} fontSize="9" fill="#555555">epoch 0</text>
              <text x={W - R} y={H - 8} fontSize="9" textAnchor="end" fill="#555555">{maxEpochs}</text>
              <text x={8} y={(H - B) / 2 + 6} fontSize="9" fill="#555555" transform={`rotate(-90 8 ${(H - B) / 2 + 6})`}>loss</text>
              {history.length > 0 && <path d={line("testLoss")} fill="none" stroke="#5B7BA8" strokeWidth="2" strokeDasharray="5 3" />}
              {history.length > 0 && <path d={line("loss")} fill="none" stroke="#BC0031" strokeWidth="2" />}
            </svg>
          );
        };

        const NeuralNetworkViz = () => {
          const canvasRef = useRef(null);
          const rngRef = useRef(makeRng(Number.isNaN(seedParam) ? Math.floor(Math.random() * 1e9) : seedParam));
          const rng = () => rngRef.current();
          const [dataKey, setDataKey] = useState(startData);
          const trainingData = DATASETS[dataKey].data;
          const [isTraining, setIsTraining] = useState(false);
          const [epoch, setEpoch] = useState(0);
          const [accuracy, setAccuracy] = useState(null);
          const [testAccuracy, setTestAccuracy] = useState(null);
          const [loss, setLoss] = useState(null);
          const [history, setHistory] = useState([]);
          const [changes, setChanges] = useState([]);
          const [learningRate, setLearningRate] = useState(0.1);
          const [currentSample, setCurrentSample] = useState(params.get("mode") === "walk" ? DATASETS[startData].data[2] : null);
          const [prediction, setPrediction] = useState(null);
          const [maxEpochs, setMaxEpochs] = useState(100);
          const [speed, setSpeed] = useState(100);
          const [weights1, setWeights1] = useState([]);
          const [weights2, setWeights2] = useState([]);
          const [biases1, setBiases1] = useState([]);
          const [biases2, setBiases2] = useState([]);
          const [mode, setMode] = useState(params.get("mode") === "walk" ? "walk" : "epochs");
          const [walkName, setWalkName] = useState("Cat 3");
          const [stage, setStage] = useState(0);
          const [walk, setWalk] = useState(null);
          const [focusH, setFocusH] = useState(null);
          const [maths, setMaths] = useState(params.get("maths") === "1");

          const initWeights = () => {
            setWeights1(Array(4).fill(0).map(() => Array(6).fill(0).map(() => (rng() - 0.5) * 0.5)));
            setWeights2(Array(6).fill(0).map(() => Array(2).fill(0).map(() => (rng() - 0.5) * 0.5)));
            setBiases1(Array(6).fill(0).map(() => (rng() - 0.5) * 0.1));
            setBiases2(Array(2).fill(0).map(() => (rng() - 0.5) * 0.1));
          };
          useEffect(initWeights, []);

          const backward = (input, target, w1, b1, w2, b2) => {
            const hiddenRaw = b1.map((bias, i) => input.reduce((acc, val, j) => acc + val * w1[j][i], 0) + bias);
            const hidden = hiddenRaw.map(relu);
            const outputRaw = b2.map((bias, i) => hidden.reduce((acc, val, j) => acc + val * w2[j][i], 0) + bias);
            const probs = softmax(outputRaw);
            const outputGrad = probs.map((p, i) => p - (i === target ? 1 : 0));
            const hiddenGrad = hidden.map((h, i) => outputGrad.reduce((acc, og, j) => acc + og * w2[i][j], 0) * reluDeriv(hiddenRaw[i]));
            return {
              w1Grad: w1.map((row, i) => row.map((_, j) => input[i] * hiddenGrad[j])),
              b1Grad: hiddenGrad,
              w2Grad: w2.map((row, i) => row.map((_, j) => hidden[i] * outputGrad[j])),
              b2Grad: outputGrad,
              loss: -Math.log(probs[target] + 1e-10),
            };
          };

          const trainStep = () => {
            if (weights1.length === 0) return;
            let totalLoss = 0;
            const shuffled = [...trainingData].sort(() => rng() - 0.5);
            const newW1 = weights1.map((row) => [...row]);
            const newB1 = [...biases1];
            const newW2 = weights2.map((row) => [...row]);
            const newB2 = [...biases2];
            shuffled.forEach((sample, idx) => {
              const g = backward(sample.features, sample.label, newW1, newB1, newW2, newB2);
              totalLoss += g.loss;
              for (let i = 0; i < 4; i++) for (let j = 0; j < 6; j++) newW1[i][j] -= learningRate * g.w1Grad[i][j];
              for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) newW2[i][j] -= learningRate * g.w2Grad[i][j];
              for (let i = 0; i < 6; i++) newB1[i] -= learningRate * g.b1Grad[i];
              for (let i = 0; i < 2; i++) newB2[i] -= learningRate * g.b2Grad[i];
              if (idx === shuffled.length - 1) {
                const r = forward(sample.features, newW1, newB1, newW2, newB2);
                setCurrentSample(sample);
                setPrediction({ probs: r.probs, predicted: r.probs[0] > r.probs[1] ? 0 : 1 });
              }
            });
            // which weights moved most in this epoch? (the heart of backpropagation)
            const moved = [];
            for (let i = 0; i < 4; i++) for (let j = 0; j < 6; j++) moved.push({ layer: 1, i, j, delta: newW1[i][j] - weights1[i][j] });
            for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) moved.push({ layer: 2, i, j, delta: newW2[i][j] - weights2[i][j] });
            moved.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
            setChanges(moved.slice(0, 5));

            const tr = evaluate(trainingData, newW1, newB1, newW2, newB2);
            const te = evaluate(TEST, newW1, newB1, newW2, newB2);
            setWeights1(newW1); setWeights2(newW2); setBiases1(newB1); setBiases2(newB2);
            setLoss(totalLoss / trainingData.length);
            setAccuracy(tr.acc);
            setTestAccuracy(te.acc);
            setHistory((h) => h.concat([{ epoch: h.length + 1, loss: totalLoss / trainingData.length, testLoss: te.loss }]));
            setEpoch((e) => e + 1);
          };

          useEffect(() => {
            if (!isTraining || weights1.length === 0) return;
            if (epoch >= maxEpochs) { setIsTraining(false); return; }
            const interval = setInterval(trainStep, speed);
            return () => clearInterval(interval);
          }, [isTraining, weights1, weights2, biases1, biases2, learningRate, epoch, maxEpochs, speed, dataKey]);

          // ---- drawing for one animal, step by step
          const AMBER = "196, 112, 0";
          const drawWalk = (ctx, width, height, layers, getY, edge) => {
            const w = walk;
            const after = stage === 5;
            const vals = after ? { x: w.x, hidden: w.after.hidden, probs: w.after.probs } : { x: w.x, hidden: w.hidden, probs: w.probs };
            const W1 = stage >= 4 ? w.nw1 : weights1, W2 = stage >= 4 ? w.nw2 : weights2;
            const font = (px, bold) => `${bold ? "600 " : ""}${px}px "Source Sans 3", Arial, sans-serif`;
            // edges
            if (stage === 4) changes.forEach((c) => edge(c.layer - 1, c.i, c.j, c.layer === 1 ? W1[c.i][c.j] : W2[c.i][c.j], true));
            if (stage === 3) {
              for (let k = 0; k < 4; k++) for (let i = 0; i < 6; i++) {
                ctx.lineWidth = 0.6; ctx.strokeStyle = "rgba(190, 198, 209, 0.6)";
                ctx.beginPath(); ctx.moveTo(layers[0].x, getY(0, k)); ctx.lineTo(layers[1].x, getY(1, i)); ctx.stroke();
              }
              for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) {
                // blame only flows into nodes that were on; edges to nodes that were off stay faint
                const f = w.hiddenRaw[i] > 0 ? w.parts[i][j] : 0, a = Math.min(Math.abs(f) * 6, 1);
                const x1 = layers[2].x, y1 = getY(2, j), x2 = layers[1].x, y2 = getY(1, i);
                ctx.lineWidth = Math.max(0.6, Math.min(Math.abs(f) * 30, 7));
                ctx.strokeStyle = `rgba(${AMBER}, ${0.15 + 0.85 * a})`;
                ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
                if (a > 0.15) {
                  // arrowhead pointing back, towards the hidden node
                  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, ang = Math.atan2(y2 - y1, x2 - x1), s = 9;
                  ctx.fillStyle = `rgba(${AMBER}, ${0.15 + 0.85 * a})`;
                  ctx.beginPath(); ctx.moveTo(mx + s * Math.cos(ang), my + s * Math.sin(ang));
                  ctx.lineTo(mx + s * Math.cos(ang + 2.5), my + s * Math.sin(ang + 2.5));
                  ctx.lineTo(mx + s * Math.cos(ang - 2.5), my + s * Math.sin(ang - 2.5)); ctx.fill();
                }
              }
            } else {
              for (let k = 0; k < 4; k++) for (let i = 0; i < 6; i++) edge(0, k, i, W1[k][i], false);
              for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) edge(1, i, j, W2[i][j], false);
            }
            // nodes with their numbers
            layers.forEach((layer, l) => {
              for (let i = 0; i < layer.nodes; i++) {
                const y = getY(l, i);
                const v = l === 0 ? vals.x[i] : l === 1 ? vals.hidden[i] : vals.probs[i];
                const ring = (stage === 3 && l === 1 && i === focusH) ? `rgb(${AMBER})` : (stage === 2 && l === 2) || (stage >= 3 && l === 1 && w.hidPush[i] !== 0 && stage === 3) ? `rgba(${AMBER}, 0.8)` : "#BEC6D1";
                ctx.fillStyle = `rgba(188, 0, 49, ${0.08 + 0.5 * Math.min(v, 1)})`;
                ctx.strokeStyle = ring; ctx.lineWidth = ring === "#BEC6D1" ? 2 : 3.5;
                ctx.beginPath(); ctx.arc(layer.x, y, 20, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                ctx.fillStyle = "#1F1D21"; ctx.font = font(11, true); ctx.textAlign = "center";
                ctx.fillText(v.toFixed(2), layer.x, y + 4);
                if (l === 1) {
                  ctx.fillStyle = "#555555"; ctx.font = font(11); ctx.fillText("H" + (i + 1), layer.x, y - 25);
                  if (stage === 3) {
                    ctx.textAlign = "left"; ctx.font = font(12, true);
                    const p = w.hidPush[i];
                    ctx.fillStyle = p === 0 ? "#8A8A8A" : `rgb(${AMBER})`;
                    ctx.fillText(p === 0 ? "off" : sgn(p), layer.x + 25, y + 4);
                  }
                }
              }
            });
            ctx.fillStyle = "#1F1D21"; ctx.font = font(13); ctx.textAlign = "right";
            FEATURES.forEach((name, i) => ctx.fillText(name, 96, getY(0, i) + 4));
            ctx.textAlign = "left";
            ["🐱 Cat", "🐶 Dog"].forEach((t, j) => {
              ctx.fillStyle = "#1F1D21"; ctx.font = font(13); ctx.fillText(t, layers[2].x + 30, getY(2, j) + 4);
              if (stage >= 2 && stage <= 3) {
                ctx.fillStyle = `rgb(${AMBER})`; ctx.font = font(12, true);
                ctx.fillText("should be " + w.target[j].toFixed(0), layers[2].x + 30, getY(2, j) + 21);
                ctx.fillText(sgn(w.outPush[j]), layers[2].x + 30, getY(2, j) + 37);
              }
            });
            ctx.textAlign = "center"; ctx.fillStyle = "#1F1D21"; ctx.font = font(13);
            ctx.fillText("Input", layers[0].x, 13); ctx.fillText("Hidden", layers[1].x, 13); ctx.fillText("Output", layers[2].x, 13);
          };

          // ---- network drawing
          useEffect(() => {
            const canvas = canvasRef.current;
            if (!canvas || weights1.length === 0) return;
            const ctx = canvas.getContext("2d");
            const width = canvas.width, height = canvas.height;
            ctx.clearRect(0, 0, width, height);
            const layers = [{ nodes: 4, x: 120 }, { nodes: 6, x: 300 }, { nodes: 2, x: 480 }];
            const getY = (l, n) => (height / (layers[l].nodes + 1)) * (n + 1);
            const edge = (l, i, j, w, halo) => {
              const x1 = layers[l].x, y1 = getY(l, i), x2 = layers[l + 1].x, y2 = getY(l + 1, j);
              if (halo) {
                ctx.lineWidth = 11; ctx.strokeStyle = "rgba(240, 196, 206, 0.95)"; ctx.lineCap = "round";
                ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
              }
              const a = Math.abs(w);
              ctx.lineWidth = Math.max(0.5, Math.min(a * 4, 4));
              ctx.strokeStyle = w > 0 ? `rgba(91, 123, 168, ${Math.min(a * 2, 1)})` : `rgba(188, 0, 49, ${Math.min(a * 2, 1)})`;
              ctx.lineCap = "butt";
              ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
            };
            if (mode === "walk" && walk && stage >= 1) {
              drawWalk(ctx, width, height, layers, getY, edge);
              return;
            }
            // halos first, underneath everything
            changes.forEach((c) => edge(c.layer - 1, c.i, c.j, c.layer === 1 ? weights1[c.i][c.j] : weights2[c.i][c.j], true));
            for (let i = 0; i < 4; i++) for (let j = 0; j < 6; j++) edge(0, i, j, weights1[i][j], false);
            for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) edge(1, i, j, weights2[i][j], false);

            const result = currentSample ? forward(currentSample.features, weights1, biases1, weights2, biases2) : null;
            layers.forEach((layer, l) => {
              for (let i = 0; i < layer.nodes; i++) {
                const y = getY(l, i);
                ctx.fillStyle = "#FFFFFF"; ctx.strokeStyle = "#BEC6D1"; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(layer.x, y, 20, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                if (result) {
                  const act = l === 0 ? currentSample.features[i] : l === 1 ? Math.min(result.hidden[i], 1) : result.probs[i];
                  ctx.fillStyle = `rgba(31, 29, 33, ${act})`;
                  ctx.beginPath(); ctx.arc(layer.x, y, 16, 0, Math.PI * 2); ctx.fill();
                }
                if (l === 1) {
                  ctx.fillStyle = "#555555"; ctx.font = '11px "Source Sans 3", Arial, sans-serif'; ctx.textAlign = "center";
                  ctx.fillText("H" + (i + 1), layer.x, y - 25);
                }
              }
            });
            ctx.fillStyle = "#1F1D21";
            ctx.font = '13px "Source Sans 3", Arial, sans-serif';
            ctx.textAlign = "right";
            FEATURES.forEach((name, i) => ctx.fillText(name, 96, getY(0, i) + 4));
            ctx.textAlign = "left";
            ctx.fillText("🐱 Cat", layers[2].x + 30, getY(2, 0) + 4);
            ctx.fillText("🐶 Dog", layers[2].x + 30, getY(2, 1) + 4);
            ctx.textAlign = "center";
            ctx.fillText("Input", layers[0].x, 13);
            ctx.fillText("Hidden", layers[1].x, 13);
            ctx.fillText("Output", layers[2].x, 13);
          }, [weights1, weights2, currentSample, changes, mode, walk, stage, focusH]);

          const walkSample = trainingData.find((s) => s.name === walkName) || trainingData[0];
          const walkNext = () => {
            if (weights1.length === 0) return;
            if (stage === 0 || stage === 5) {
              let sample = walkSample;
              if (stage === 5) {
                const k = trainingData.indexOf(walkSample);
                sample = trainingData[(k + 1) % trainingData.length];
                setWalkName(sample.name);
              }
              const w = walkThrough(sample, weights1, biases1, weights2, biases2, learningRate);
              setWalk(w); setStage(1); setChanges([]);
              const hs = w.hidPush.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]);
              setFocusH(hs[0][1]);
              setCurrentSample(sample);
              setPrediction({ probs: w.probs, predicted: w.probs[0] > w.probs[1] ? 0 : 1 });
              return;
            }
            if (stage === 3) {
              // the update really happens: the network has now learned from this one animal
              setWeights1(walk.nw1); setWeights2(walk.nw2); setBiases1(walk.nb1); setBiases2(walk.nb2);
              setChanges(walk.top.map((c) => ({ layer: c.layer, i: c.i, j: c.j, delta: c.delta })));
              const tr = evaluate(trainingData, walk.nw1, walk.nb1, walk.nw2, walk.nb2);
              const te = evaluate(TEST, walk.nw1, walk.nb1, walk.nw2, walk.nb2);
              setAccuracy(tr.acc); setTestAccuracy(te.acc); setLoss(tr.loss);
            }
            if (stage === 4) setPrediction({ probs: walk.after.probs, predicted: walk.after.probs[0] > walk.after.probs[1] ? 0 : 1 });
            setStage(stage + 1);
          };
          const walkRestart = (name) => {
            if (name) setWalkName(name);
            setStage(0); setWalk(null); setChanges([]); setFocusH(null);
            const s = trainingData.find((t) => t.name === (name || walkName)) || trainingData[0];
            setCurrentSample(s); setPrediction(null);
          };
          const switchMode = (m) => {
            setIsTraining(false); setMode(m); setStage(0); setWalk(null); setChanges([]); setFocusH(null);
            if (m === "walk") { setCurrentSample(walkSample); setPrediction(null); }
          };
          const canvasClick = (e) => {
            if (mode !== "walk" || stage < 3 || !walk) return;
            const r = e.currentTarget.getBoundingClientRect();
            const cx = ((e.clientX - r.left) / r.width) * 600, cy = ((e.clientY - r.top) / r.height) * 400;
            for (let i = 0; i < 6; i++) {
              const y = (400 / 7) * (i + 1);
              if (Math.hypot(cx - 300, cy - y) < 26) { setFocusH(i); return; }
            }
          };

          const reset = (key) => {
            setIsTraining(false);
            setStage(0); setWalk(null); setFocusH(null);
            setEpoch(0); setAccuracy(null); setTestAccuracy(null); setLoss(null);
            setHistory([]); setChanges([]); setCurrentSample(null); setPrediction(null);
            if (typeof key === "string") setDataKey(key);
            initWeights();
          };

          const edgeName = (c) => c.layer === 1 ? `${FEATURES[c.i]} → H${c.j + 1}` : `H${c.i + 1} → ${c.j === 0 ? "Cat" : "Dog"}`;
          const ready = weights1.length > 0;
          const verdicts = (set) => set.map((s) => {
            const p = ready ? forward(s.features, weights1, biases1, weights2, biases2).probs : [0.5, 0.5];
            return { ...s, pCat: p[0], right: (p[0] > p[1] ? 0 : 1) === s.label };
          });
          const pct = (v) => (v == null ? "–" : v.toFixed(1) + "%");

          const AnimalRow = ({ s }) => (
            <div className="flex items-center gap-2 text-sm py-1 border-b cda-rule">
              <span className="w-20 shrink-0">{s.label === 0 ? "🐱" : "🐶"} {s.name}</span>
              <div className="flex-1 cda-track rounded-full h-2" title={`P(cat) = ${(s.pCat * 100).toFixed(0)}%`}>
                <div className="cda-bar h-2 rounded-full" style={{ width: `${s.pCat * 100}%` }} />
              </div>
              <span className="font-mono w-10 text-right">{(s.pCat * 100).toFixed(0)}%</span>
              <span className={`w-5 text-center font-bold ${epoch === 0 ? "cda-faint" : s.right ? "cda-good" : "cda-bad"}`}>{epoch === 0 ? "·" : s.right ? "✓" : "✗"}</span>
              <span className="cda-faint text-xs w-36 truncate">{s.note || ""}</span>
            </div>
          );

          const pc = (v) => (v * 100).toFixed(1) + "%";
          const WalkPanel = () => {
            const w = walk;
            const name = walkSample.name;
            const isCat = walkSample.label === 0;
            const right = isCat ? "Cat" : "Dog";
            let body;
            if (stage === 0 || !w) {
              body = <p>Follow one training animal through one round of backpropagation. Choose an animal and press <b>Start</b>.
                Each press of <b>Next</b> shows one stage. This is exactly what happens 12 times in every epoch.
                <span className="cda-faint"> (Rumelhart, Hinton and Williams 1986)</span></p>;
            } else if (stage === 1) {
              body = <div>
                <p>The four numbers of {name} go in on the left. Each hidden node adds them up, each multiplied by its weight, and passes the
                  result on (or 0, if the sum is below zero: the node is <i>off</i>). The output turns the result into two probabilities.</p>
                <p className="mt-2 font-semibold cda-ink">The network says: cat {pc(w.probs[0])}, dog {pc(w.probs[1])}.</p>
              </div>;
            } else if (stage === 2) {
              const o = isCat ? 0 : 1;
              body = <div>
                <p>The right answer is <b>{right}</b>: cat should be {w.target[0]}, dog should be {w.target[1]}.
                  The difference is the error.</p>
                <div className="text-sm mt-2 font-mono">
                  <div className="flex justify-between"><span>Cat: {w.target[0]} − {w.probs[0].toFixed(2)}</span><span className="wk-amber font-bold">{sgn(w.outPush[0])}</span></div>
                  <div className="flex justify-between"><span>Dog: {w.target[1]} − {w.probs[1].toFixed(2)}</span><span className="wk-amber font-bold">{sgn(w.outPush[1])}</span></div>
                </div>
                <p className="mt-2">So {right.toLowerCase()} should go up by {Math.abs(w.outPush[o]).toFixed(2)}, and the other output down by the same amount.</p>
              </div>;
            } else if (stage === 3) {
              const i = focusH == null ? 0 : focusH;
              const off = w.hiddenRaw[i] <= 0;
              body = <div>
                <p>The error now goes back. Each hidden node gets a share of the blame (in technical terms, the <i>gradient</i>):
                  how much higher or lower it should have been. A node gets more blame when its connection to an output is strong.</p>
                <div className="text-sm mt-2 p-2 border rounded cda-rule" style={{ padding: "8px 10px" }} data-k="blame-sum">
                  <div className="font-semibold mb-1">H{i + 1}{off ? " (off)" : ""}</div>
                  {off ? <div>H{i + 1} was off (0) for {name}, so it gets no blame. Its weights will not change this time.</div> : <div>
                    <div className="cda-muted mb-1">weight × error at the output, for both outputs:</div>
                    <div className="font-mono">
                      <div>→ Cat: {sgn(w.w2[i][0])} × {sgn(w.outPush[0])} = {sgn(w.parts[i][0], 3)}</div>
                      <div>→ Dog: {sgn(w.w2[i][1])} × {sgn(w.outPush[1])} = {sgn(w.parts[i][1], 3)}</div>
                    </div>
                    <div className="border-t cda-rule mt-1 pt-2 font-bold wk-amber">H{i + 1} should change by {sgn(w.hidPush[i], 3)}</div>
                  </div>}
                </div>
                <p className="text-xs cda-faint mt-2">Click another hidden node in the picture to see its sum.</p>
              </div>;
            } else if (stage === 4) {
              const t = w.top[0];
              body = <div>
                <p>Now every weight changes a little, by three things multiplied together: the learning rate, how active the node at its
                  start was, and the blame at its end. Nodes that were off change nothing.</p>
                <div className="text-sm mt-2 p-2 border rounded cda-rule font-mono" style={{ padding: "8px 10px" }} data-k="update-sum">
                  <div className="font-semibold mb-1" style={{ fontFamily: "inherit" }}>{t.fromName} → {t.toName}</div>
                  <div>{w.lr.toFixed(2)} × {t.from.toFixed(2)} × {sgn(t.push, 3)} = {sgn(t.delta, 3)}</div>
                  <div>{t.before.toFixed(3)} → {(t.before + t.delta).toFixed(3)}</div>
                </div>
                <div className="text-sm mt-2">
                  {w.top.map((c, k) => (
                    <div key={k} className="flex justify-between border-b cda-rule py-1">
                      <span>{c.fromName} → {c.toName}</span>
                      <span className={`font-mono ${c.delta > 0 ? "cda-pos" : "cda-bad"}`}>{sgn(c.delta, 3)}</span>
                    </div>
                  ))}
                </div>
              </div>;
            } else {
              const up = w.after.probs[walkSample.label] > w.probs[walkSample.label];
              body = <div>
                <p>The same animal again, with the new weights:</p>
                <p className="mt-2 font-semibold cda-ink" data-k="before-after">{right}: {pc(w.probs[walkSample.label])} → {pc(w.after.probs[walkSample.label])}</p>
                <p className="mt-2">{up ? "One small step in the right direction." : "Not better this time."} Training repeats this for every animal,
                  epoch after epoch. Press <b>Next animal</b> to go on, or switch to <b>Whole epochs</b> to watch many steps add up.</p>
              </div>;
            }
            let m = null;
            if (maths && w && stage >= 1) {
              const L = (p) => -Math.log(p + 1e-10);
              const g = (v) => sgn(-v, 3); // gradient = minus the push
              const i = focusH == null ? 0 : focusH;
              const t = w.top[0];
              if (stage === 1) m = <div><div>z = Σ w·x + b,  h = ReLU(z) = max(0, z)</div><div>p = softmax(Σ w·h + b)</div></div>;
              if (stage === 2) m = <div>
                <div>L = −log p(right answer) = {L(w.probs[walkSample.label]).toFixed(3)}</div>
                <div>∂L/∂z = p − y:  cat {g(w.outPush[0])}, dog {g(w.outPush[1])}</div>
                <div className="wk-note">Shown above with the sign flipped: “should go up”.</div></div>;
              if (stage === 3) m = <div>
                <div>∂L/∂h{i + 1} = (Σ w · ∂L/∂z) · ReLU′(z{i + 1})</div>
                <div>= ({g(w.parts[i][0] + w.parts[i][1])}) × {w.hiddenRaw[i] > 0 ? 1 : 0} = {g(w.hidPush[i])}</div>
                <div className="wk-note">The chain rule. ReLU′ is 1 for a node that was on, 0 for a node that was off.</div></div>;
              if (stage === 4) m = <div>
                <div>Δw = −η · ∂L/∂w = −η · x · ∂L/∂z</div>
                <div>= −{w.lr.toFixed(2)} × {t.from.toFixed(2)} × ({g(t.push)}) = {sgn(t.delta, 3)}</div>
                <div className="wk-note">η is the learning rate; x is the activation at the start of the weight. One step of gradient descent.</div></div>;
              if (stage === 5) m = <div><div>L: {L(w.probs[walkSample.label]).toFixed(3)} → {L(w.after.probs[walkSample.label]).toFixed(3)}</div></div>;
            }
            return (
              <div className="cda-panel rounded p-4" data-k="walk-panel">
                <div className="text-xs cda-muted">{stage === 0 ? "One animal at a time" : `Stage ${stage} of 5`}</div>
                <h3 className="text-lg font-semibold mb-2">{STAGES[stage]}</h3>
                <div className="text-sm">{body}</div>
                {m && <div className="wk-maths font-mono text-xs mt-3" data-k="maths-box">{m}</div>}
              </div>
            );
          };

          return (
            <div className="w-full min-h-screen cda-page p-6">
              <div className="max-w-6xl mx-auto">
                <h1 className="text-3xl font-bold mb-2 text-center">Neural Network Backpropagation</h1>
                <p className="cda-muted cda-sub text-center mb-6">Cat vs Dog Classification with Weight Updates</p>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  <div className="cda-panel rounded p-4">
                    <h3 className="text-lg font-semibold mb-3">Training Progress</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between"><span className="cda-muted">Epoch:</span><span className="font-mono font-bold cda-ink">{epoch} / {maxEpochs}</span></div>
                      <div className="flex justify-between"><span className="cda-muted">Accuracy, training animals:</span><span className="font-mono font-bold cda-good" data-k="train-acc">{pct(accuracy)}</span></div>
                      <div className="flex justify-between"><span className="cda-muted">Accuracy, 4 unseen animals:</span><span className="font-mono font-bold cda-pos" data-k="test-acc">{pct(testAccuracy)}</span></div>
                      <div className="flex justify-between"><span className="cda-muted">Loss:</span><span className="font-mono font-bold cda-bad" data-k="loss">{loss == null ? "–" : loss.toFixed(4)}</span></div>
                      {epoch >= maxEpochs && <div className="cda-bad text-center pt-2 border-t cda-rule">✓ Training Complete</div>}
                    </div>
                  </div>

                  <div className="cda-panel rounded p-4">
                    <h3 className="text-lg font-semibold mb-3">Current Sample</h3>
                    {currentSample ? (
                      <div className="space-y-2">
                        <div className="text-xl text-center mb-2">{currentSample.label === 0 ? "🐱" : "🐶"} {currentSample.name}</div>
                        <div className="text-sm cda-muted">
                          {FEATURES.map((name, i) => (
                            <div key={i} className="flex justify-between"><span>{name}:</span><span className="font-mono">{currentSample.features[i].toFixed(2)}</span></div>
                          ))}
                        </div>
                      </div>
                    ) : (<div className="cda-faint text-center py-4">Start training to see samples</div>)}
                  </div>

                  <div className="cda-panel rounded p-4">
                    <h3 className="text-lg font-semibold mb-3">Prediction</h3>
                    {prediction ? (
                      <div className="space-y-2">
                        <div className="flex justify-between items-center mb-2"><span>🐱 Cat:</span><span className="font-mono font-bold">{(prediction.probs[0] * 100).toFixed(1)}%</span></div>
                        <div className="w-full cda-track rounded-full h-2"><div className="cda-bar h-2 rounded-full transition-all duration-300" style={{ width: `${prediction.probs[0] * 100}%` }} /></div>
                        <div className="flex justify-between items-center mb-2 mt-3"><span>🐶 Dog:</span><span className="font-mono font-bold">{(prediction.probs[1] * 100).toFixed(1)}%</span></div>
                        <div className="w-full cda-track rounded-full h-2"><div className="cda-bar h-2 rounded-full transition-all duration-300" style={{ width: `${prediction.probs[1] * 100}%` }} /></div>
                        <div className="mt-3 text-center">
                          <span className={`font-bold ${prediction.predicted === currentSample?.label ? "cda-good" : "cda-bad"}`}>
                            {prediction.predicted === currentSample?.label ? "✓ Correct" : "✗ Incorrect"}
                          </span>
                        </div>
                      </div>
                    ) : (<div className="cda-faint text-center py-4">No prediction yet</div>)}
                  </div>
                </div>

                <div className="cda-panel rounded p-4 mb-6">
                  <div className="flex flex-wrap items-center gap-2 text-sm mb-3 pb-3 border-b cda-rule" role="group" aria-label="View">
                    <span className="cda-muted">Show:</span>
                    <button data-k="mode-epochs" onClick={() => switchMode("epochs")} aria-pressed={mode === "epochs"}
                      className={`px-3 py-1 rounded-lg ${mode === "epochs" ? "cda-btn-primary" : "cda-btn"}`}>Whole epochs</button>
                    <button data-k="mode-walk" onClick={() => switchMode("walk")} aria-pressed={mode === "walk"}
                      className={`px-3 py-1 rounded-lg ${mode === "walk" ? "cda-btn-primary" : "cda-btn"}`}>One animal at a time</button>
                  </div>
                  <div className="flex flex-wrap gap-4 items-center justify-between">
                    {mode === "epochs" ? (
                    <div className="flex gap-2">
                      <button onClick={() => setIsTraining(!isTraining)} disabled={epoch >= maxEpochs} className="flex items-center gap-2 cda-btn-primary px-4 py-2 rounded-lg transition-colors">
                        {isTraining ? "⏸" : "▶"} {isTraining ? "Pause" : "Train"}
                      </button>
                      <button id="step-button" onClick={trainStep} disabled={isTraining || epoch >= maxEpochs} className="flex items-center gap-2 cda-btn px-4 py-2 rounded-lg transition-colors">⚡ Step</button>
                      <button id="reset-button" onClick={() => reset()} className="flex items-center gap-2 cda-btn px-4 py-2 rounded-lg transition-colors">↻ Reset</button>
                    </div>
                    ) : (
                    <div className="flex flex-wrap gap-2 items-center">
                      <label className="text-sm cda-muted" htmlFor="walk-animal">Animal:</label>
                      <select id="walk-animal" value={walkSample.name} onChange={(e) => walkRestart(e.target.value)} className="bg-white border cda-rule rounded px-2 py-1 text-sm">
                        {trainingData.map((t) => <option key={t.name} value={t.name}>{t.label === 0 ? "🐱" : "🐶"} {t.name}</option>)}
                      </select>
                      <button id="step-button" onClick={walkNext} className="flex items-center gap-2 cda-btn-primary px-4 py-2 rounded-lg transition-colors">
                        {stage === 0 ? "Start ▶" : stage === 5 ? "Next animal ▶" : "Next ▶"}
                      </button>
                      <button id="reset-button" onClick={() => reset()} className="flex items-center gap-2 cda-btn px-4 py-2 rounded-lg transition-colors">↻ Reset</button>
                      <label className="flex items-center gap-2 text-sm cda-muted" style={{ cursor: "pointer" }}>
                        <input type="checkbox" data-k="maths" checked={maths} onChange={(e) => setMaths(e.target.checked)} style={{ accentColor: "#BC0031" }} />
                        Show the maths
                      </label>
                    </div>
                    )}
                    <div className="flex items-center gap-2 text-sm" role="group" aria-label="Training data">
                      <span className="cda-muted">Data:</span>
                      {Object.entries(DATASETS).map(([k, d]) => (
                        <button key={k} data-k={"data-" + k} onClick={() => reset(k)} aria-pressed={dataKey === k}
                          className={`px-3 py-1 rounded-lg ${dataKey === k ? "cda-btn-primary" : "cda-btn"}`}>{d.label}</button>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-6 mt-4">
                    <div className="flex items-center gap-3">
                      <label className="text-sm cda-muted">Learning Rate:</label>
                      <input type="range" min="0.01" max="0.5" step="0.01" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} className="w-32" />
                      <span className="font-mono text-sm w-12">{learningRate.toFixed(2)}</span>
                    </div>
                    {mode === "epochs" && <div className="flex items-center gap-3">
                      <label className="text-sm cda-muted">Max Epochs:</label>
                      <input type="number" min="10" max="1000" step="10" value={maxEpochs} onChange={(e) => setMaxEpochs(parseInt(e.target.value))} className="w-20 bg-white border cda-rule rounded px-2 py-1 text-sm" />
                    </div>}
                    {mode === "epochs" && <div className="flex items-center gap-3">
                      <label className="text-sm cda-muted">Speed:</label>
                      <input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(parseInt(e.target.value))} className="w-32" />
                      <span className="font-mono text-sm w-16">{(1000 / speed).toFixed(1)}/s</span>
                    </div>}
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  <div className="cda-panel rounded p-4 lg:col-span-2">
                    <canvas ref={canvasRef} width={600} height={400} className="w-full rounded" onClick={canvasClick}
                      style={{ background: "#FFFFFF", cursor: mode === "walk" && stage === 3 ? "pointer" : "default" }} />
                    {mode === "walk" && stage >= 1 ? (
                    <div className="mt-4 text-sm cda-muted text-center">
                      Numbers in the circles = how active each node is for {walk.sample.name} ·
                      {stage === 3 ? <span> <span className="wk-amber font-semibold">amber</span> = the blame flowing back; thicker = more blame. Click a hidden node to see its sum.</span>
                        : stage === 4 ? <span> <span className="font-semibold cda-ink"><span className="halo-key">pink halo</span></span> = the 5 weights that moved most for this animal</span>
                        : <span> <span className="cda-pos font-semibold">blue-grey</span> = positive weight, <span className="cda-bad font-semibold">crimson</span> = negative</span>}
                    </div>
                    ) : (
                    <div className="mt-4 text-sm cda-muted text-center">
                      <span className="cda-pos font-semibold">Blue-grey</span> = positive weight ·
                      <span className="cda-bad font-semibold"> crimson</span> = negative · thickness = strength ·
                      <span className="font-semibold cda-ink"> <span className="halo-key">pink halo</span></span> = the 5 weights that changed most in the last epoch
                    </div>
                    )}
                  </div>
                  <div className="flex flex-col gap-6">
                    {mode === "walk" && <WalkPanel />}
                    <div className="cda-panel rounded p-4">
                      <h3 className="text-lg font-semibold mb-1">Loss per epoch</h3>
                      <LossChart history={history} maxEpochs={maxEpochs} />
                      <div className="text-xs cda-muted flex gap-4 mt-1">
                        <span><span className="cda-bad font-bold">━</span> training animals</span>
                        <span><span className="cda-pos font-bold">- -</span> unseen animals</span>
                      </div>
                    </div>
                    {mode === "walk" ? null : <div className="cda-panel rounded p-4">
                      <h3 className="text-lg font-semibold mb-1">Biggest weight changes</h3>
                      {changes.length ? (
                        <div className="text-sm" data-k="changes">
                          {changes.map((c, k) => (
                            <div key={k} className="flex justify-between border-b cda-rule py-1">
                              <span>{edgeName(c)}</span>
                              <span className={`font-mono ${c.delta > 0 ? "cda-pos" : "cda-bad"}`}>{c.delta > 0 ? "+" : "−"}{Math.abs(c.delta).toFixed(3)}</span>
                            </div>
                          ))}
                          <p className="cda-faint text-xs mt-2">Backpropagation sends the error back through the network and changes each weight by how much it contributed.</p>
                        </div>
                      ) : (<div className="cda-faint text-sm py-2">Press Step to see which weights the error changes most.</div>)}
                    </div>}
                  </div>
                </div>

                <div className="cda-panel rounded p-4 mb-6">
                  <h3 className="text-lg font-semibold mb-1">Every animal, right now</h3>
                  <p className="text-sm cda-muted mb-3">The bar is the network's probability for <b>cat</b>. ✓ = sorted correctly.</p>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                      <div className="text-sm font-semibold mb-1">Training animals ({trainingData.length})</div>
                      {verdicts(trainingData).map((s) => <AnimalRow key={s.name} s={s} />)}
                    </div>
                    <div>
                      <div className="text-sm font-semibold mb-1">Unseen animals (never trained on)</div>
                      {verdicts(TEST).map((s) => <AnimalRow key={s.name} s={s} />)}
                    </div>
                  </div>
                </div>

                <div className="cda-panel rounded p-4 text-sm cda-muted">
                  <h3 className="font-semibold cda-ink mb-2">How it works:</h3>
                  <p>
                    A small neural network learns to sort cats from dogs by backpropagation, using four made-up features.
                    In each epoch it sees every training animal once, measures its error, and nudges every weight in the direction that
                    reduces the error; the pink halos show which weights moved most. The overlapping data include a cat with folded ears,
                    a dog with pointy ears, and a cat and a dog with exactly the same numbers: no network can get both of those right.
                    The four unseen animals are never trained on, so they show whether the network has learned something that carries over.
                  </p>
                  <p className="mt-2">
                    <b>One animal at a time</b> slows a single update down into five stages: the network guesses, the error is measured at the
                    output, the blame flows back to the hidden nodes, every weight takes a small step, and the network guesses again.
                  </p>
                  <p className="mt-2">
                    Backpropagation: David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams, “Learning representations by
                    back-propagating errors”, <i>Nature</i> 323 (1986), 533–536. <a href="https://doi.org/10.1038/323533a0" target="_blank" rel="noopener" className="cda-bad" style={{ textDecoration: "underline" }}>doi.org/10.1038/323533a0</a>
                  </p>
                </div>
              </div>
            </div>
          );
        };

        ReactDOM.render(<NeuralNetworkViz />, document.getElementById("root"));
