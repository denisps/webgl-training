// EfficientWord-Net inspired embedding model using WebGL dense layers.
//
// The architecture mirrors the EfficientWord-Net pipeline:
//   1. Input:  64-dim time-averaged log mel features (computed in audio.js)
//   2. Dense(64 → 128, relu)
//   3. Dense(128 → 64, relu)  ← L2-normalised output used as embedding
//   4. Head(64 → 2, linear)   ← only used during cross-entropy training
//
// Detection compares embeddings with cosine similarity.
// Training uses cross-entropy: label 1 = hotword, label 0 = background.

import { createTensor, destroyTensor, readTensor } from '../../src/webgl/context.js';
import { denseFwd, denseBwd, createDenseWeights } from '../../src/layers/dense.js';

export const N_MEL = 64;
const D_HIDDEN = 128;
const D_EMBED = 64;
const N_CLASSES = 2;

function flattenWeights(weights) {
  return [
    weights.dense1.W, weights.dense1.b,
    weights.dense2.W, weights.dense2.b,
    weights.head.W,   weights.head.b,
  ];
}

export function efficientWordNetFwd(gl, programs, input, weights) {
  const layer1 = denseFwd(gl, programs, input, weights.dense1, 'relu');
  const layer2 = denseFwd(gl, programs, layer1.output, weights.dense2, 'relu');
  const head   = denseFwd(gl, programs, layer2.output, weights.head, 'linear');
  return { output: head.output, cache: { layer1, layer2, head } };
}

export function efficientWordNetBwd(gl, programs, input, weights, cache, dOutput) {
  const gHead = denseBwd(gl, programs, cache.layer2.output, weights.head, cache.head.preActivation, dOutput, 'linear');
  const g2    = denseBwd(gl, programs, cache.layer1.output, weights.dense2, cache.layer2.preActivation, gHead.dInput, 'relu');
  const g1    = denseBwd(gl, programs, input, weights.dense1, cache.layer1.preActivation, g2.dInput, 'relu');
  destroyTensor(gl, gHead.dInput);
  destroyTensor(gl, g2.dInput);
  return {
    dInput: g1.dInput,
    gradients: [g1.dW, g1.db, g2.dW, g2.db, gHead.dW, gHead.db],
  };
}

function disposeCache(gl, cache) {
  destroyTensor(gl, cache.layer1.output);
  destroyTensor(gl, cache.layer1.preActivation);
  destroyTensor(gl, cache.layer2.output);
  destroyTensor(gl, cache.layer2.preActivation);
  destroyTensor(gl, cache.head.preActivation);
}

export function createEfficientWordNetModel(gl) {
  const weights = {
    dense1: createDenseWeights(gl, N_MEL,    D_HIDDEN),
    dense2: createDenseWeights(gl, D_HIDDEN,  D_EMBED),
    head:   createDenseWeights(gl, D_EMBED,   N_CLASSES),
  };
  return {
    type:          'efficientwordnet',
    inputSize:     N_MEL,
    embeddingSize: D_EMBED,
    weights,
    parameters: flattenWeights(weights),
    forward:      (gl, programs, input, w) => efficientWordNetFwd(gl, programs, input, w),
    backward:     (gl, programs, input, w, cache, dOut) => efficientWordNetBwd(gl, programs, input, w, cache, dOut),
    disposeCache: (gl, cache) => disposeCache(gl, cache),
  };
}

// Run input through Dense1 → Dense2 and return L2-normalised D_EMBED-dim embedding.
export function extractEmbedding(gl, programs, model, melFeatures) {
  const inputTensor = createTensor(gl, 1, N_MEL, new Float32Array(melFeatures));
  const layer1 = denseFwd(gl, programs, inputTensor, model.weights.dense1, 'relu');
  const layer2 = denseFwd(gl, programs, layer1.output, model.weights.dense2, 'relu');
  const raw = readTensor(gl, layer2.output);
  destroyTensor(gl, inputTensor);
  destroyTensor(gl, layer1.output);
  destroyTensor(gl, layer1.preActivation);
  destroyTensor(gl, layer2.output);
  destroyTensor(gl, layer2.preActivation);
  return l2Normalize(raw);
}

export function l2Normalize(vec) {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm) + 1e-8;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

// Cosine similarity between two L2-normalised vectors
export function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// Maximum cosine similarity between a query embedding and a set of reference embeddings
export function maxCosineSimilarity(queryEmbedding, referenceEmbeddings) {
  let max = -1;
  for (const ref of referenceEmbeddings) {
    const sim = cosineSimilarity(queryEmbedding, ref);
    if (sim > max) max = sim;
  }
  return max;
}

// Serialise only the backbone weights (Dense1 + Dense2) for the reference file.
// The classifier head is not needed for inference.
export function serializeBackboneWeights(gl, model) {
  return [
    { rows: model.weights.dense1.W.rows, cols: model.weights.dense1.W.cols, data: Array.from(readTensor(gl, model.weights.dense1.W)) },
    { rows: model.weights.dense1.b.rows, cols: model.weights.dense1.b.cols, data: Array.from(readTensor(gl, model.weights.dense1.b)) },
    { rows: model.weights.dense2.W.rows, cols: model.weights.dense2.W.cols, data: Array.from(readTensor(gl, model.weights.dense2.W)) },
    { rows: model.weights.dense2.b.rows, cols: model.weights.dense2.b.cols, data: Array.from(readTensor(gl, model.weights.dense2.b)) },
  ];
}

// Restore backbone weights from a serialised array (as produced by serializeBackboneWeights).
export function loadBackboneWeights(gl, model, weightsData) {
  const fields = [
    ['dense1', 'W'],
    ['dense1', 'b'],
    ['dense2', 'W'],
    ['dense2', 'b'],
  ];
  for (let i = 0; i < 4; i++) {
    const [layer, key] = fields[i];
    const { rows, cols, data } = weightsData[i];
    destroyTensor(gl, model.weights[layer][key]);
    model.weights[layer][key] = createTensor(gl, rows, cols, new Float32Array(data));
  }
  model.parameters = flattenWeights(model.weights);
}
