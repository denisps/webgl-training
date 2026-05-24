import { destroyTensor } from '../../src/webgl/context.js';
import { denseBwd, denseFwd, createDenseWeights } from '../../src/layers/dense.js';

function flattenWeights(weights) {
  return [
    weights.l1.W, weights.l1.b,
    weights.l2.W, weights.l2.b,
    weights.l3.W, weights.l3.b,
  ];
}

export function mapModelFwd(gl, programs, input, weights) {
  const l1 = denseFwd(gl, programs, input, weights.l1, 'relu');
  const l2 = denseFwd(gl, programs, l1.output, weights.l2, 'relu');
  const l3 = denseFwd(gl, programs, l2.output, weights.l3, 'linear');
  return {
    output: l3.output,
    cache: { l1, l2, l3 },
  };
}

export function mapModelBwd(gl, programs, input, weights, cache, dOutput) {
  const g3 = denseBwd(gl, programs, cache.l2.output, weights.l3, cache.l3.preActivation, dOutput, 'linear');
  const g2 = denseBwd(gl, programs, cache.l1.output, weights.l2, cache.l2.preActivation, g3.dInput, 'relu');
  const g1 = denseBwd(gl, programs, input, weights.l1, cache.l1.preActivation, g2.dInput, 'relu');
  destroyTensor(gl, g3.dInput);
  destroyTensor(gl, g2.dInput);
  return {
    dInput: g1.dInput,
    gradients: [g1.dW, g1.db, g2.dW, g2.db, g3.dW, g3.db],
  };
}

export function createMapModel(gl, nRegions = 8) {
  const weights = {
    l1: createDenseWeights(gl, 32 * 32 * 3, 512),
    l2: createDenseWeights(gl, 512, 128),
    l3: createDenseWeights(gl, 128, nRegions),
  };
  const layers = [
    { ...weights.l1, activation: 'relu' },
    { ...weights.l2, activation: 'relu' },
    { ...weights.l3, activation: 'linear' },
  ];
  return {
    type: 'dense',
    inputSize: 32 * 32 * 3,
    layers,
    weights,
    parameters: flattenWeights(weights),
    architecture: {
      type: 'dense',
      nRegions,
      layers: [
        { inFeatures: 32 * 32 * 3, outFeatures: 512, activation: 'relu' },
        { inFeatures: 512, outFeatures: 128, activation: 'relu' },
        { inFeatures: 128, outFeatures: nRegions, activation: 'linear' },
      ],
    },
    forward: mapModelFwd,
    backward: mapModelBwd,
    disposeCache(glContext, cache) {
      [
        cache.l1.output,
        cache.l1.preActivation,
        cache.l2.output,
        cache.l2.preActivation,
        cache.l3.preActivation,
      ].forEach((tensor) => destroyTensor(glContext, tensor));
    },
  };
}
