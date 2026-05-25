import { createTensor, destroyTensor, readTensor } from '../webgl/context.js';
import { denseFwd } from '../layers/dense.js';
import { destroyTransformerCache, transformerFwd } from '../layers/transformer.js';

export function runInference(gl, programs, model, input) {
  if (model.type === 'dense') {
    let current = input;
    let owned = false;
    for (const layer of model.layers) {
      const result = denseFwd(gl, programs, current, layer, layer.activation || 'linear');
      destroyTensor(gl, result.preActivation);
      if (owned) {
        destroyTensor(gl, current);
      }
      current = result.output;
      owned = true;
    }
    return current;
  }

  if (model.type === 'transformer') {
    let current = input;
    let owned = false;
    for (const block of model.blocks) {
      const result = transformerFwd(gl, programs, current, block, model.architecture.nHeads);
      if (owned) {
        destroyTensor(gl, current);
      }
      destroyTransformerCache(gl, result.cache);
      current = result.output;
      owned = true;
    }
    return current;
  }

  // Generic fallback: models that expose a forward function and disposeCache.
  if (typeof model.forward === 'function') {
    const { output, cache } = model.forward(gl, programs, input, model.weights);
    model.disposeCache(gl, cache);
    return output;
  }

  throw new Error(`Unsupported model type: ${model.type}`);
}

export function runBatchInference(gl, programs, model, inputs) {
  const cols = model.inputSize;
  const rows = inputs.length / cols;
  const inputTensor = createTensor(gl, rows, cols, inputs);
  const outputTensor = runInference(gl, programs, model, inputTensor);
  const output = readTensor(gl, outputTensor);
  destroyTensor(gl, inputTensor);
  destroyTensor(gl, outputTensor);
  return output;
}
