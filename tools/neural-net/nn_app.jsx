
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
          const [currentSample, setCurrentSample] = useState(null);
          const [prediction, setPrediction] = useState(null);
          const [maxEpochs, setMaxEpochs] = useState(100);
          const [speed, setSpeed] = useState(100);
          const [weights1, setWeights1] = useState([]);
          const [weights2, setWeights2] = useState([]);
          const [biases1, setBiases1] = useState([]);
          const [biases2, setBiases2] = useState([]);

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
          }, [weights1, weights2, currentSample, changes]);

          const reset = (key) => {
            setIsTraining(false);
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
                  <div className="flex flex-wrap gap-4 items-center justify-between">
                    <div className="flex gap-2">
                      <button onClick={() => setIsTraining(!isTraining)} disabled={epoch >= maxEpochs} className="flex items-center gap-2 cda-btn-primary px-4 py-2 rounded-lg transition-colors">
                        {isTraining ? "⏸" : "▶"} {isTraining ? "Pause" : "Train"}
                      </button>
                      <button onClick={trainStep} disabled={isTraining || epoch >= maxEpochs} className="flex items-center gap-2 cda-btn px-4 py-2 rounded-lg transition-colors">⚡ Step</button>
                      <button onClick={() => reset()} className="flex items-center gap-2 cda-btn px-4 py-2 rounded-lg transition-colors">↻ Reset</button>
                    </div>
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
                    <div className="flex items-center gap-3">
                      <label className="text-sm cda-muted">Max Epochs:</label>
                      <input type="number" min="10" max="1000" step="10" value={maxEpochs} onChange={(e) => setMaxEpochs(parseInt(e.target.value))} className="w-20 bg-white border cda-rule rounded px-2 py-1 text-sm" />
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-sm cda-muted">Speed:</label>
                      <input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(parseInt(e.target.value))} className="w-32" />
                      <span className="font-mono text-sm w-16">{(1000 / speed).toFixed(1)}/s</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  <div className="cda-panel rounded p-4 lg:col-span-2">
                    <canvas ref={canvasRef} width={600} height={400} className="w-full rounded" style={{ background: "#FFFFFF" }} />
                    <div className="mt-4 text-sm cda-muted text-center">
                      <span className="cda-pos font-semibold">Blue-grey</span> = positive weight ·
                      <span className="cda-bad font-semibold"> crimson</span> = negative · thickness = strength ·
                      <span className="font-semibold cda-ink"> <span className="halo-key">pink halo</span></span> = the 5 weights that changed most in the last epoch
                    </div>
                  </div>
                  <div className="flex flex-col gap-6">
                    <div className="cda-panel rounded p-4">
                      <h3 className="text-lg font-semibold mb-1">Loss per epoch</h3>
                      <LossChart history={history} maxEpochs={maxEpochs} />
                      <div className="text-xs cda-muted flex gap-4 mt-1">
                        <span><span className="cda-bad font-bold">━</span> training animals</span>
                        <span><span className="cda-pos font-bold">- -</span> unseen animals</span>
                      </div>
                    </div>
                    <div className="cda-panel rounded p-4">
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
                    </div>
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
                </div>
              </div>
            </div>
          );
        };

        ReactDOM.render(<NeuralNetworkViz />, document.getElementById("root"));
