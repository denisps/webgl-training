import { createTensor, destroyTensor } from '../webgl/context.js';
import { matmul, matmulAt, matmulBt } from '../ops/matmul.js';
import { addBias, geluBwd, geluFwd, reluBwd, reluFwd, scale, sigmoidBwd, sigmoidFwd } from '../ops/elementwise.js';
import { sumCols } from '../ops/reduce.js';

export const DENSE_SHADERS = {};

function xavierData(size, fanIn, fanOut) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  const data = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    data[index] = Math.random() * 2 * limit - limit;
  }
  return data;
}

export function createDenseWeights(gl, inFeatures, outFeatures) {
  return {
    W: createTensor(gl, inFeatures, outFeatures, xavierData(inFeatures * outFeatures, inFeatures, outFeatures)),
    b: createTensor(gl, 1, outFeatures, new Float32Array(outFeatures)),
  };
}

function applyActivation(gl, programs, tensor, activation) {
  switch (activation) {
    case 'relu':
      return reluFwd(gl, programs, tensor);
    case 'gelu':
      return geluFwd(gl, programs, tensor);
    case 'sigmoid':
      return sigmoidFwd(gl, programs, tensor);
    case 'linear':
    default:
      return scale(gl, programs, tensor, 1.0);
  }
}

function activationBackward(gl, programs, preActivation, dOutput, activation) {
  switch (activation) {
    case 'relu':
      return reluBwd(gl, programs, preActivation, dOutput);
    case 'gelu':
      return geluBwd(gl, programs, preActivation, dOutput);
    case 'sigmoid':
      return sigmoidBwd(gl, programs, preActivation, dOutput);
    case 'linear':
    default:
      return scale(gl, programs, dOutput, 1.0);
  }
}

export function denseFwd(gl, programs, input, weights, activation = 'linear') {
  const linear = matmul(gl, programs, input, weights.W);
  const preActivation = addBias(gl, programs, linear, weights.b);
  const output = applyActivation(gl, programs, preActivation, activation);
  destroyTensor(gl, linear);
  return { output, preActivation };
}

export function denseBwd(gl, programs, input, weights, preActivation, dOutput, activation = 'linear') {
  const dZ = activationBackward(gl, programs, preActivation, dOutput, activation);
  const dW = matmulAt(gl, programs, input, dZ);
  const db = sumCols(gl, programs, dZ);
  const dInput = matmulBt(gl, programs, dZ, weights.W);
  return { dInput, dW, db };
}
