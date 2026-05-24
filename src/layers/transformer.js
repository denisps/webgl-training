import { createTensor, destroyTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';
import { matmul, matmulAt, matmulBt } from '../ops/matmul.js';
import { add, scale } from '../ops/elementwise.js';
import { layerNormBwd, layerNormFwd } from '../ops/norm.js';
import { scaledDotProductBwd, scaledDotProductFwd } from '../ops/attention.js';
import { denseBwd, denseFwd } from './dense.js';

export const TRANSFORMER_SHADERS = {
  sliceCols: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_X;
uniform int u_Start;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  outColor = texelFetch(u_X, ivec2(col + u_Start, row), 0).r;
}`,
  concat2: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_A;
uniform sampler2D u_B;
uniform int u_Split;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  if (col < u_Split) {
    outColor = texelFetch(u_A, ivec2(col, row), 0).r;
  } else {
    outColor = texelFetch(u_B, ivec2(col - u_Split, row), 0).r;
  }
}`,
};

function xavierTensor(gl, rows, cols) {
  const limit = Math.sqrt(6 / (rows + cols));
  const data = new Float32Array(rows * cols);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = Math.random() * 2 * limit - limit;
  }
  return createTensor(gl, rows, cols, data);
}

function zerosTensor(gl, rows, cols) {
  return createTensor(gl, rows, cols, new Float32Array(rows * cols));
}

function onesTensor(gl, rows, cols) {
  const data = new Float32Array(rows * cols);
  data.fill(1);
  return createTensor(gl, rows, cols, data);
}

function sliceCols(gl, programs, tensor, start, width) {
  const output = createTensor(gl, tensor.rows, width);
  executePass(gl, programs.sliceCols, {
    u_X: tensor.texture,
    u_Start: start,
  }, output, programs.__quadBuffer);
  return output;
}

function concat2(gl, programs, A, B) {
  const output = createTensor(gl, A.rows, A.cols + B.cols);
  executePass(gl, programs.concat2, {
    u_A: A.texture,
    u_B: B.texture,
    u_Split: A.cols,
  }, output, programs.__quadBuffer);
  return output;
}

function concatMany(gl, programs, tensors) {
  if (tensors.length === 1) {
    return scale(gl, programs, tensors[0], 1.0);
  }
  let current = concat2(gl, programs, tensors[0], tensors[1]);
  for (let index = 2; index < tensors.length; index += 1) {
    const next = concat2(gl, programs, current, tensors[index]);
    destroyTensor(gl, current);
    current = next;
  }
  return current;
}

export function createTransformerWeights(gl, dModel, nHeads, ffnDim) {
  if (dModel % nHeads !== 0) {
    throw new Error('dModel must be divisible by nHeads.');
  }
  return {
    Wq: xavierTensor(gl, dModel, dModel),
    Wk: xavierTensor(gl, dModel, dModel),
    Wv: xavierTensor(gl, dModel, dModel),
    Wo: xavierTensor(gl, dModel, dModel),
    W1: xavierTensor(gl, dModel, ffnDim),
    b1: zerosTensor(gl, 1, ffnDim),
    W2: xavierTensor(gl, ffnDim, dModel),
    b2: zerosTensor(gl, 1, dModel),
    gamma1: onesTensor(gl, 1, dModel),
    beta1: zerosTensor(gl, 1, dModel),
    gamma2: onesTensor(gl, 1, dModel),
    beta2: zerosTensor(gl, 1, dModel),
  };
}

export function transformerFwd(gl, programs, input, weights, nHeads) {
  const headDim = input.cols / nHeads;
  const ln1 = layerNormFwd(gl, programs, input, weights.gamma1, weights.beta1, 1e-5);
  const Q = matmul(gl, programs, ln1.output, weights.Wq);
  const K = matmul(gl, programs, ln1.output, weights.Wk);
  const V = matmul(gl, programs, ln1.output, weights.Wv);

  const heads = [];
  for (let head = 0; head < nHeads; head += 1) {
    const start = head * headDim;
    const q = sliceCols(gl, programs, Q, start, headDim);
    const k = sliceCols(gl, programs, K, start, headDim);
    const v = sliceCols(gl, programs, V, start, headDim);
    const attn = scaledDotProductFwd(gl, programs, q, k, v);
    heads.push({ q, k, v, ...attn });
  }

  const concatInput = heads.map((head) => head.output);
  const concat = concatMany(gl, programs, concatInput);
  const attnProj = matmul(gl, programs, concat, weights.Wo);
  const res1 = add(gl, programs, input, attnProj);

  const ln2 = layerNormFwd(gl, programs, res1, weights.gamma2, weights.beta2, 1e-5);
  const ff1 = denseFwd(gl, programs, ln2.output, { W: weights.W1, b: weights.b1 }, 'gelu');
  const ff2 = denseFwd(gl, programs, ff1.output, { W: weights.W2, b: weights.b2 }, 'linear');
  const output = add(gl, programs, res1, ff2.output);

  return {
    output,
    cache: {
      ln1,
      Q,
      K,
      V,
      heads,
      concat,
      attnProj,
      res1,
      ln2,
      ff1,
      ff2,
    },
  };
}

export function transformerBwd(gl, programs, input, weights, cache, dOutput, nHeads) {
  const dFf2 = scale(gl, programs, dOutput, 1.0);
  let dRes1 = scale(gl, programs, dOutput, 1.0);

  const ff2Grads = denseBwd(gl, programs, cache.ff1.output, { W: weights.W2, b: weights.b2 }, cache.ff2.preActivation, dFf2, 'linear');
  const ff1Grads = denseBwd(gl, programs, cache.ln2.output, { W: weights.W1, b: weights.b1 }, cache.ff1.preActivation, ff2Grads.dInput, 'gelu');
  const ln2Grads = layerNormBwd(gl, programs, cache.res1, weights.gamma2, cache.ln2.mean, cache.ln2.invStd, ff1Grads.dInput);
  const dRes1Total = add(gl, programs, dRes1, ln2Grads.dX);
  destroyTensor(gl, dRes1);
  dRes1 = dRes1Total;

  const dAttnProj = scale(gl, programs, dRes1, 1.0);
  const dConcat = matmulBt(gl, programs, dAttnProj, weights.Wo);
  const dWo = matmulAt(gl, programs, cache.concat, dAttnProj);

  const headDim = input.cols / nHeads;
  const dQHeads = [];
  const dKHeads = [];
  const dVHeads = [];
  for (let head = 0; head < nHeads; head += 1) {
    const start = head * headDim;
    const dHeadOut = sliceCols(gl, programs, dConcat, start, headDim);
    const headCache = cache.heads[head];
    const grads = scaledDotProductBwd(gl, programs, headCache.q, headCache.k, headCache.v, headCache.attnWeights, dHeadOut);
    dQHeads.push(grads.dQ);
    dKHeads.push(grads.dK);
    dVHeads.push(grads.dV);
    destroyTensor(gl, dHeadOut);
  }
  destroyTensor(gl, dConcat);

  const dQ = concatMany(gl, programs, dQHeads);
  const dK = concatMany(gl, programs, dKHeads);
  const dV = concatMany(gl, programs, dVHeads);

  const dWq = matmulAt(gl, programs, cache.ln1.output, dQ);
  const dWk = matmulAt(gl, programs, cache.ln1.output, dK);
  const dWv = matmulAt(gl, programs, cache.ln1.output, dV);
  const dLn1Q = matmulBt(gl, programs, dQ, weights.Wq);
  const dLn1K = matmulBt(gl, programs, dK, weights.Wk);
  const dLn1V = matmulBt(gl, programs, dV, weights.Wv);
  const dLn1Tmp = add(gl, programs, dLn1Q, dLn1K);
  const dLn1 = add(gl, programs, dLn1Tmp, dLn1V);

  destroyTensor(gl, dQ);
  destroyTensor(gl, dK);
  destroyTensor(gl, dV);
  destroyTensor(gl, dLn1Q);
  destroyTensor(gl, dLn1K);
  destroyTensor(gl, dLn1V);
  destroyTensor(gl, dLn1Tmp);

  const ln1Grads = layerNormBwd(gl, programs, input, weights.gamma1, cache.ln1.mean, cache.ln1.invStd, dLn1);
  const dInput = add(gl, programs, dRes1, ln1Grads.dX);

  return {
    dInput,
    dWeights: {
      Wq: dWq,
      Wk: dWk,
      Wv: dWv,
      Wo: dWo,
      W1: ff1Grads.dW,
      b1: ff1Grads.db,
      W2: ff2Grads.dW,
      b2: ff2Grads.db,
      gamma1: ln1Grads.dGamma,
      beta1: ln1Grads.dBeta,
      gamma2: ln2Grads.dGamma,
      beta2: ln2Grads.dBeta,
    },
  };
}

export function destroyTransformerCache(gl, cache) {
  const tensors = [
    cache.ln1.output,
    cache.ln1.mean,
    cache.ln1.invStd,
    cache.Q,
    cache.K,
    cache.V,
    cache.concat,
    cache.attnProj,
    cache.res1,
    cache.ln2.output,
    cache.ln2.mean,
    cache.ln2.invStd,
    cache.ff1.output,
    cache.ff1.preActivation,
    cache.ff2.output,
    cache.ff2.preActivation,
  ];
  for (const head of cache.heads) {
    tensors.push(head.q, head.k, head.v, head.output, head.attnWeights);
  }
  for (const tensor of tensors) {
    destroyTensor(gl, tensor);
  }
}
