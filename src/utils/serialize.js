import { readTensor } from '../webgl/context.js';

export function serializeWeights(gl, weightTensors) {
  const serialized = weightTensors.map((tensor) => ({
    rows: tensor.rows,
    cols: tensor.cols,
    data: Array.from(readTensor(gl, tensor)),
  }));
  return JSON.stringify(serialized, null, 2);
}

export function deserializeWeights(json) {
  const parsed = typeof json === 'string' ? JSON.parse(json) : json;
  return parsed.map((entry) => ({
    rows: entry.rows,
    cols: entry.cols,
    data: new Float32Array(entry.data),
  }));
}

export function downloadWeights(gl, weightTensors, filename = 'weights.json') {
  const blob = new Blob([serializeWeights(gl, weightTensors)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

// Save model weights together with any metadata (e.g. architecture) so the
// file is self-contained for later inference. The saved format is:
//   { ...metadata, weights: [{rows, cols, data}, ...] }
// loadModelWeights() in inference/model.js already reads this format.
export function downloadModel(gl, metadata, weightTensors, filename = 'model.json') {
  const weights = weightTensors.map((tensor) => ({
    rows: tensor.rows,
    cols: tensor.cols,
    data: Array.from(readTensor(gl, tensor)),
  }));
  const blob = new Blob([JSON.stringify({ ...metadata, weights }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function loadWeightsFromFile() {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json,.json';
    input.onchange = async () => {
      const [file] = input.files || [];
      if (!file) {
        resolve(null);
        return;
      }
      try {
        const text = await file.text();
        resolve(JSON.parse(text));
      } catch (error) {
        reject(error);
      }
    };
    input.click();
  });
}
