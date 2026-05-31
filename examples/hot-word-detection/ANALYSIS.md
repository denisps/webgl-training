# Hot Word Detection: Analysis and Comparison with EfficientWord-Net

## 1. How This Implementation Relates to EfficientWord-Net

The hot-word detection example is **inspired by** EfficientWord-Net in its interface design and audio pipeline parameters, but shares none of its model weights, architecture depth, or training dataset. The connection is:

- Identical audio hyperparameters: 16 kHz, 25 ms frames, 10 ms hop, 512-point FFT, 64 mel bins, no pre-emphasis — lifted verbatim from the `Resnet50_Arc_loss.compute_logfbank_features` call.
- Same user-facing workflow: record 4–10 pronunciations → generate a `_ref.json` reference file → run inference with cosine similarity against stored embeddings.
- Same 1.5-second sliding window and threshold-based triggering.

Everything else — the neural network, how it is trained, the dataset, and the weights — is entirely different.

---

## 2. Dataset

| | EfficientWord-Net | This implementation |
|---|---|---|
| Pre-training dataset | MLCommons Multilingual Spoken Words corpus (100k+ speakers, 50 languages) | None |
| Fine-tuning / per-user data | Not supported | 4–10 user recordings per hotword |
| Background samples | Baked into the pre-trained model | User-recorded or procedurally generated |

EfficientWord-Net ships with a fully pre-trained ONNX model; users never touch the training data. This implementation trains a tiny MLP from scratch on the handful of samples the user records in-browser.

---

## 3. Pre-trained Weights

EfficientWord-Net bundles two ONNX models:

- `slim_93%_accuracy_72.7390%.onnx` — full precision ResNet-50 + ArcFace (~88 MB)
- `slim_93%_accuracy_72.7390%_qint8.onnx` — int8 quantized version

These are frozen; the library never modifies them. Users produce per-hotword embedding files (`*_ref.json`) by running the reference-generation script, which feeds their recordings through the frozen backbone and stores the resulting embeddings.

This implementation uses Xavier-initialised random weights and has no pre-training of any kind. Everything is learned from the user's own recordings.

---

## 4. Model Architecture

### EfficientWord-Net (current: `Resnet50_Arc_loss`)

```
Input: logfbank features  [148 × 64]  (NUMFRAMES × N_MEL)
       reshaped to [1 × 1 × 148 × 64]  (batch × channel × H × W)

Backbone: ResNet-50
  - Initial 7×7 conv, BatchNorm, ReLU, MaxPool
  - 4 residual stage groups (3-4-6-3 blocks each)
  - Each block: 1×1 conv → 3×3 conv → 1×1 conv + skip connection
  - Global Average Pooling → 2048-dim vector
  
ArcFace head (training only):
  - Normalised linear layer → logits per class
  - Angular margin softmax loss
  
Output at inference: 2048-dim L2-normalised embedding

Detection: max cosine similarity vs stored reference embeddings
```

### EfficientWord-Net (original: `First_Iteration_Siamese`, deprecated)

```
Siamese twin CNN network trained with triplet loss.
Input: same mel spectrogram; comparing pairs of recordings.
Removed in v1.0.3.
```

### This Implementation (`efficientWordNet` in model.js)

```
Input: time-averaged logfbank features  [1 × 64]  (single vector)

Dense(64 → 128, ReLU)       ~8 k parameters
Dense(128 → 64, ReLU)       ~8 k parameters   ← L2-normalised as embedding
Dense(64 → 2, linear)       ~130 parameters    ← classification head

Total: ~16 k parameters (vs ~25 M for ResNet-50)

Loss: cross-entropy (hotword=1, background=0)
Optimizer: Adam, all layers trained jointly
```

The critical structural difference is that this MLP collapses the entire 1.5-second spectrogram to a single 64-dim vector **before the network sees it**, discarding all temporal information. EfficientWord-Net's ResNet-50 processes the full 148×64 spectrogram as a 2D image, allowing it to learn time-frequency patterns.

---

## 5. Layer-Type Comparison

| Component | EfficientWord-Net | This Implementation |
|---|---|---|
| Input features | 148×64 2D spectrogram | 64-dim time-averaged vector |
| First layer | 7×7 Conv2D + BN + ReLU | Dense(64→128, ReLU) |
| Core layers | 16× bottleneck residual blocks | Dense(128→64, ReLU) |
| Normalisation | BatchNorm after every conv | None |
| Residual connections | Yes | No |
| Pooling | Global Average Pooling | N/A |
| Embedding dim | 2048 | 64 |
| Training loss | ArcFace (angular margin softmax) | Cross-entropy |
| Activation | ReLU throughout | ReLU throughout |

---

## 6. What EfficientWord-Net Tests Can Be Ported

EfficientWord-Net's test files (`test.py`, `test_inference.py`) are purely **integration / live tests** — they open the microphone and print detection results. There are no offline unit tests or accuracy benchmarks in the public repository.

The following EfficientWord-Net testing patterns can be reproduced here:

### 6.1 Offline single-hotword detection test
Record a set of utterances offline, run each through the model, and assert that hotword utterances score above threshold and non-hotword utterances score below.

```js
// pseudocode
const scores = hotwordSamples.map(audio => classifyMel(gl, programs, model, extractLogMelFeatures(audio)));
assert(scores.every(s => s > threshold));
const falsePositives = backgroundSamples.map(audio => classifyMel(...));
assert(falsePositives.every(s => s < threshold));
```

### 6.2 False-accept rate on background speech
Play arbitrary speech through the model and count triggers per hour. EfficientWord-Net targets < 1 false accept per hour.

### 6.3 Relaxation-time / debounce test
Feed the same hotword in rapid succession and verify the detector only fires once within the relaxation window. The current implementation lacks this; see improvement notes below.

### 6.4 Multi-hotword discrimination
Two reference sets for two different words — verify that scoring a frame with word A returns a higher similarity for A's embeddings than B's.

### 6.5 Embedding consistency test
Extract the embedding of a hotword sample twice (identical audio). Assert cosine similarity ≈ 1.0 (checks determinism).

---

## 7. Improving Hot-Word Detection

### 7.1 Preserve temporal structure (highest impact)

The single largest accuracy gain would come from using the full 148×64 log-mel spectrogram as input instead of the time-averaged 64-dim vector. Temporal patterns (onset, vowel, consonant transitions) carry most of the word-identity information.

Options in increasing complexity:

**Option A — 1D dense on stacked frames (simplest)**
```
Input: [148 × 64] flatten → [9472-dim]
Dense(9472 → 256, ReLU) → Dense(256 → 64, ReLU) → Head(64 → 2)
```
Works but is parameter-heavy and sensitive to timing shifts.

**Option B — 1D convolutions over time**
```
Input: [148 × 64] treated as 148 timesteps of 64 features
Conv1D(64 → 64, kernel=5, stride=2) × 3 → Global pool → Dense → Head
```
More time-shift invariant. Requires implementing a Conv1D layer (map to matmul with `im2col` in a shader).

**Option C — Small transformer (already in framework)**
```
Input: [148 × 64] (148 tokens, 64-dim each)
TransformerBlock × 2 → CLS token → Head
```
`src/layers/transformer.js` already exists. This is the highest quality option.

### 7.2 Switch to metric learning (embedding-only training)

Cross-entropy on 2 classes forces the model to learn a classification boundary rather than a general embedding space. Replacing with contrastive or triplet loss would make the embedding more generalisable:

- **Contrastive loss**: push same-class embeddings together, pull different-class apart.
- **Triplet loss**: anchor (hotword), positive (another hotword sample), negative (background) — minimise anchor-positive distance, maximise anchor-negative distance.
- **ArcFace**: like cross-entropy but with an angular margin, producing better-separated embeddings.

This is how EfficientWord-Net achieves good few-shot accuracy from only 4 samples at inference time.

### 7.3 Add relaxation time / debounce

The current detection loop triggers on every 1.5 s window where confidence exceeds threshold. Add a cooldown:

```js
let lastDetectionTime = 0;
const RELAXATION_MS = 800;
// inside detection loop:
if (confidence > threshold && (Date.now() - lastDetectionTime) > RELAXATION_MS) {
  lastDetectionTime = Date.now();
  // fire event
}
```

### 7.4 Voice Activity Detection (VAD)

Skip model inference when there is no speech. A simple RMS gate is already implemented (`computeRms` in `audio.js`), but a proper VAD would use a per-frame energy profile or zero-crossing rate to reduce false positives during silence or music.

### 7.5 Better data augmentation

Current augmentations: time shift ±100 ms, gain 0.7–1.3×, additive noise 0–1.5 %. Additional useful augmentations:

- **SpecAugment**: zero out random time and frequency bands directly on the mel spectrogram.
- **Room impulse response convolution**: simulate different recording environments.
- **Background mixing**: overlay hotword with background audio at random SNR.

### 7.6 More negative samples

The current approach uses 3–8 user-recorded non-hotword phrases plus synthetic background. The biggest source of false positives is speech with a similar phonetic structure. Recording the full alphabet, common words, and phonetically similar words improves the classifier's decision boundary.

---

## 8. Moving CPU Computations to Shaders

The entire audio feature extraction pipeline (`audio.js`) runs on the CPU. Below is a breakdown of each step and its GPU migration path.

### 8.1 Hamming window application

**Current:** Per-sample multiply in JS.  
**GPU version:** A simple fragment shader reading from a time-domain audio texture and multiplying by a pre-loaded Hamming texture. Trivial to implement as a `mul` pass using the existing `ELEMENTWISE_SHADERS`.

### 8.2 FFT

**Current:** Cooley-Tukey radix-2 FFT in JS (~150 LOC).  
**GPU version:** GPU FFT is non-trivial. The standard approach is a multi-pass Stockham FFT (no bit-reversal permutation needed):

```glsl
// Each pass processes butterfly operations at a given stage.
// ~log2(N) = 9 passes for N=512.
// Requires a ping-pong texture strategy.
```

This would give the biggest throughput gain for continuous detection (eliminating the ~0.5–1 ms FFT per frame × 148 frames = ~100–150 ms per detection window).

An alternative is using `WebAudio` — see section 9.

### 8.3 Mel filterbank application

**Current:** `N_MEL × HALF_BINS` multiply-accumulate in JS (64 × 257 = 16 448 operations per frame × 148 frames = ~2.4 M ops per window).  
**GPU version:** Treat the power spectrum as a 1×257 texture and the filterbank as a 64×257 matrix. One `matmul` pass using existing `MATMUL_SHADERS` produces the 64-dim mel vector for a single frame. Running all 148 frames together as a batch (148×257 input) makes it a single `matmul(148×257, 257×64) → 148×64` call.

This is the most straightforward GPU migration and would be the highest-impact single change.

### 8.4 Log transform

**Current:** `Math.log(Math.max(energy, 1e-10))` per element.  
**GPU version:** A trivial `log` unary shader (not yet in `ELEMENTWISE_SHADERS` but trivial to add).

### 8.5 Time averaging

**Current:** Sum and divide in JS.  
**GPU version:** An existing `reduce` pass (`sumRows` or similar over the time axis of the 148×64 tensor) followed by a `scale` shader.

### 8.6 L2 normalisation

**Current:** JS loop in `l2Normalize`.  
**GPU version:** Requires a two-pass reduce (sum of squares → sqrt), then a divide. The `REDUCE_SHADERS` file already has `softmaxFwd` as a reference for per-row reductions; L2 norm is similar.

### 8.7 Summary: proposed GPU pipeline

```
AudioWorklet → Float32 audio chunk (CPU)
  ↓ createTensor(1, WINDOW_SAMPLES)            [upload once per window]
  ↓ hammingWindowShader(audio, hammingWeights)  [elementwise mul]
  ↓ stockhamFFT(windowed frames)               [multi-pass, log2(512)=9 passes]
  ↓ powerSpectrum(complex FFT output)           [|re|² + |im|² elementwise]
  ↓ matmul(powerSpec[148×257], melBank[257×64]) [existing matmul shader]
  ↓ logShader(melEnergies)                      [unary log]
  ↓ sumRows + scale                             [time-average → 1×64]
  ↓ denseFwd × 3                               [model forward pass]
  ↓ readTensor(head.output)                    [readback ~8 floats]
```

The FFT is the hardest piece. Everything else maps directly to existing shader infrastructure.

---

## 9. Using WebAudio for FFT and Costly Computations

### 9.1 What WebAudio can provide

The `AnalyserNode` exposes a real-time FFT computed natively by the browser's audio engine:

```js
const analyser = audioContext.createAnalyser();
analyser.fftSize = 512;                // must be power of 2, 32–32768
analyser.smoothingTimeConstant = 0;   // no temporal smoothing
source.connect(analyser);

const powerSpectrum = new Float32Array(analyser.frequencyBinCount); // 257 bins
analyser.getFloatFrequencyData(powerSpectrum); // returns dBFS values
// Convert: power = 10^(dB/10)
```

**Pros:**
- Native performance, hardware-accelerated by the browser.
- No JS FFT code to maintain.
- Correctly windowed (uses a Blackman window by default; configurable via `AnalyserNode`).

**Cons:**
- Returns **magnitude spectrum in dBFS**, not raw power spectrum. Must convert back to linear scale.
- `getFloatFrequencyData` applies the `smoothingTimeConstant` — must be set to 0 to disable.
- The window function is fixed (Blackman-Harris, not Hamming). The difference is audible in spectrograms but small enough not to significantly affect detection accuracy in practice.
- Timing: `getFloatFrequencyData` captures the most recent frame at the moment of the call; you cannot request per-frame spectra retroactively for a batch. This makes batch mel feature extraction harder.
- The current architecture collects 148 frames and then processes the whole window — incompatible with pull-based `AnalyserNode` reads.

### 9.2 Recommended integration

For **continuous real-time detection** (one model call per sliding window), a workable hybrid is:

1. Use an `AudioWorklet` to deliver raw 128-sample chunks (as now).
2. Simultaneously connect the same source to an `AnalyserNode` configured for per-frame spectra.
3. On every worklet chunk, also call `analyser.getFloatFrequencyData()` and accumulate 148 frames' worth of mel features.
4. Run the mel filterbank (on GPU) and model forward pass when the window is full.

For **offline / training feature extraction** (where audio is captured and then processed), keep the custom JS FFT — it runs on a buffer, not a live stream, so `AnalyserNode` does not help.

The cleanest architecture splits the pipeline at the power spectrum level:

```
Live path:  AnalyserNode → getFloatFrequencyData → dBFS→power → mel filterbank (GPU)
Offline path: custom FFT (JS or GPU) → power → mel filterbank (GPU)
```

---

## 10. Testing the Correctness of the NN Framework

The existing tests in `tests/index.html` verify individual ops (matmul, relu, softmax, attention, Adam step, cross-entropy) against manually computed expected values. This is necessary but not sufficient. The following additional tests would increase confidence in the full backward pass.

### 10.1 Numerical gradient check (most important)

For any differentiable function `f(θ)`, the finite-difference gradient should match the analytical gradient:

$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon}$$

where $\epsilon \approx 10^{-3}$ is appropriate for float32.

```js
async function gradCheck(gl, programs, model, input, label, paramIndex, epsilon = 1e-3) {
  const param = model.parameters[paramIndex];
  const data = readTensor(gl, param);

  function lossAt(delta, idx) {
    const perturbed = data.slice();
    perturbed[idx] += delta;
    destroyTensor(gl, model.parameters[paramIndex]);
    model.parameters[paramIndex] = createTensor(gl, param.rows, param.cols, perturbed);
    // ... sync model.weights to match ... (needs bookkeeping)
    const { output } = model.forward(gl, programs, input, model.weights);
    const { loss } = crossEntropyLoss(gl, programs, output, [label]);
    return loss;
  }

  const numericalGrad = [];
  for (let i = 0; i < Math.min(data.length, 10); i++) {
    numericalGrad.push((lossAt(epsilon, i) - lossAt(-epsilon, i)) / (2 * epsilon));
  }
  return numericalGrad;
}
```

Compare the first 10 elements of `numericalGrad` against the first 10 elements of the gradient tensor returned by `model.backward(...)`. Agreement within 1e-2 (given float32 precision) confirms the backward pass is correct.

### 10.2 Loss decreases on a small synthetic dataset

Train for 20 epochs on a tiny XOR-like problem (4 samples, 2 classes). The loss must strictly decrease. This tests the full train loop: forward → loss → backward → Adam → forward.

```js
await runTest('testTrainingConverges', async () => {
  const data = [
    { input: [1, 0, 0, ...], label: 1 },
    { input: [0, 1, 0, ...], label: 1 },
    { input: [0, 0, 1, ...], label: 0 },
    { input: [0, 0, 0, ...], label: 0 },
  ];
  // train 30 epochs
  const lossAfter = trainEpoch(...);
  assert(lossAfter < 0.2, 'Model did not converge');
});
```

### 10.3 Gradient flow through activation functions

For each activation (relu, gelu, sigmoid), verify that `bwd(x, dy)` is zero where the forward is blocked (e.g., ReLU at x < 0) and equals `dy` where the derivative is 1.

```js
await runTest('testReluBwdSaturation', () => {
  const x   = createTensor(gl, 1, 3, new Float32Array([-5, 0.001, 5]));
  const dy  = createTensor(gl, 1, 3, new Float32Array([1, 1, 1]));
  const dx  = reluBwd(gl, programs, x, dy);
  assertArrayClose(readTensor(gl, dx), new Float32Array([0, 1, 1]));
});
```

### 10.4 Adam: verify bias correction and direction

Already partially tested. A stronger test:

```js
await runTest('testAdamBiasCorrection', () => {
  // For a constant gradient g, after t steps, the update per step should approach -lr * g / sqrt(g²)
  // i.e. the sign of the parameter change should match the sign of -g.
  // Also check that momentum causes the effective step to be lr * beta1^t * ... corrected.
});
```

### 10.5 Dense layer: weight gradient matches `input^T @ dZ`

For a single forward pass followed by a backward pass, check that `dW` equals `input.T @ dZ` computed with the reference `matmulAt` op.

### 10.6 Cross-entropy: verify gradient at uniform predictions

At `logits = [0, 0]`, `softmax = [0.5, 0.5]`. For `label = 1`, `dLogits = [0.5, -0.5]`. This is already tested. Add a batched version and verify per-sample gradients are divided by batch size correctly.

### 10.7 Serialise–deserialise round-trip

Already tested via `testSerialize`. A stronger version: train for 5 epochs, serialise, deserialise, train 5 more epochs, and verify the loss continues declining rather than jumping back up.

---

## 11. Summary Table

| Dimension | EfficientWord-Net | This Implementation |
|---|---|---|
| Model | ResNet-50 (25 M params) | 3-layer MLP (16 k params) |
| Input features | 148×64 full spectrogram | 64-dim time-averaged vector |
| Training dataset | MLCommons (100k+ speakers) | 4–10 user recordings |
| Training loss | ArcFace (metric learning) | Cross-entropy (classification) |
| Pre-trained weights | Yes (shipped ONNX) | No |
| Per-user training | No (embedding-only) | Yes (full fine-tune) |
| Audio processing | CPU (numpy / python-speech-features) | CPU (custom JS FFT) |
| Inference runtime | ONNX Runtime (CPU) | WebGL2 fragment shaders |
| Embedding dim | 2048 | 64 |
| Platform | Python / Raspberry Pi | Browser (WebGL2) |
| Window function | Rectangular (default) | Hamming |
| Pre-emphasis | 0.0 (explicitly disabled) | 0.0 |
| Relaxation time | Yes (configurable) | Not yet implemented |
| Multi-hotword | Yes | Not yet implemented |
