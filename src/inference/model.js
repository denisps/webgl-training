import { createTensor, destroyTensor } from '../webgl/context.js';
import { createDenseWeights } from '../layers/dense.js';
import { createTransformerWeights } from '../layers/transformer.js';

export const MODEL_SCHEMAS = {
  imageClassifier: {
    type: 'dense',
    inputSize: 784,
    layers: [
      { inFeatures: 784, outFeatures: 256, activation: 'relu' },
      { inFeatures: 256, outFeatures: 64, activation: 'relu' },
    ],
  },
  mapClassifier: {
    type: 'dense',
    inputSize: 3072,
    layers: [
      { inFeatures: 3072, outFeatures: 512, activation: 'relu' },
      { inFeatures: 512, outFeatures: 128, activation: 'relu' },
    ],
  },
  transformerBlock: {
    type: 'transformer',
    dModel: 32,
    nHeads: 4,
    ffnDim: 64,
    layers: [{ kind: 'transformer' }],
  },
};

function flattenDenseWeights(layers) {
  const tensors = [];
  for (const layer of layers) {
    tensors.push(layer.W, layer.b);
  }
  return tensors;
}

function flattenTransformerWeights(blocks) {
  const keys = ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'b1', 'W2', 'b2', 'gamma1', 'beta1', 'gamma2', 'beta2'];
  const tensors = [];
  for (const block of blocks) {
    for (const key of keys) {
      tensors.push(block[key]);
    }
  }
  return tensors;
}

export function createInferenceModel(gl, architecture) {
  if (architecture.type === 'dense') {
    const layers = architecture.layers.map(({ inFeatures, outFeatures, activation }) => ({
      ...createDenseWeights(gl, inFeatures, outFeatures),
      activation,
    }));
    return {
      type: 'dense',
      architecture,
      layers,
      parameters: flattenDenseWeights(layers),
      inputSize: architecture.inputSize ?? architecture.layers[0].inFeatures,
    };
  }
  if (architecture.type === 'transformer') {
    const blocks = architecture.layers.map(() => createTransformerWeights(gl, architecture.dModel, architecture.nHeads, architecture.ffnDim));
    return {
      type: 'transformer',
      architecture,
      blocks,
      parameters: flattenTransformerWeights(blocks),
      inputSize: architecture.dModel,
    };
  }
  throw new Error(`Unsupported architecture type: ${architecture.type}`);
}

export function loadModelWeights(gl, model, weightsJson) {
  const parsed = typeof weightsJson === 'string' ? JSON.parse(weightsJson) : weightsJson;
  const entries = Array.isArray(parsed) ? parsed : parsed.weights;
  if (!Array.isArray(entries) || entries.length !== model.parameters.length) {
    throw new Error('Weight JSON does not match model parameter count.');
  }
  for (let index = 0; index < entries.length; index += 1) {
    const target = model.parameters[index];
    const source = entries[index];
    if (source.rows !== target.rows || source.cols !== target.cols) {
      throw new Error(`Shape mismatch at parameter ${index}.`);
    }
    const replacement = createTensor(gl, source.rows, source.cols, new Float32Array(source.data));
    destroyTensor(gl, target);
    target.texture = replacement.texture;
  }
}
