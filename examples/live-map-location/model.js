import { createTensor, destroyTensor } from '../../src/webgl/context.js';
import { denseBwd, denseFwd, createDenseWeights } from '../../src/layers/dense.js';
import { createTransformerWeights, batchedTransformerFwd, batchedTransformerBwd, destroyTransformerCache } from '../../src/layers/transformer.js';
import { patchify, unpatchify, meanPool, meanPoolBwd, addPosEmbed, posEmbGrad } from '../../src/ops/patch.js';

// ViT-style regression model with learnable positional encoding.
// Outputs [latNorm, lonNorm, scaleNorm], all in [0,1] relative to map bounds / training range.
export const REGION_SIZE = 64;
export const OUTPUT_DIM  = 3;   // [latNorm, lonNorm, scaleNorm]

const PATCH_SIDE  = 8;
const CHANNELS    = 3;
const SEQ_LEN     = (REGION_SIZE / PATCH_SIDE) ** 2;  // 64 tokens for 64×64
const GRID_COLS   = REGION_SIZE / PATCH_SIDE;           // 8
const TOKEN_DIM   = PATCH_SIDE * PATCH_SIDE * CHANNELS; // 192
const D_MODEL     = 256;
const N_HEADS     = 4;
const FFN_DIM     = 512;
const N_LAYERS    = 2;

const TX_WEIGHT_KEYS = ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'b1', 'W2', 'b2', 'gamma1', 'beta1', 'gamma2', 'beta2'];

function flattenWeights(weights) {
  return [
    weights.embed.W, weights.embed.b,
    weights.pos,
    ...weights.tx.flatMap((tx) => TX_WEIGHT_KEYS.map((k) => tx[k])),
    weights.head.W, weights.head.b,
  ];
}

export function geoRegressionModelFwd(gl, programs, input, weights) {
  // 1. Patchify: [batch, inputSize] → [batch×SEQ_LEN, TOKEN_DIM]
  const patched = patchify(gl, programs, input, SEQ_LEN, GRID_COLS, PATCH_SIDE, REGION_SIZE, CHANNELS);
  // 2. Linear embedding: → [batch×SEQ_LEN, D_MODEL]
  const embedded = denseFwd(gl, programs, patched, weights.embed, 'linear');
  // 3. Add learnable positional encoding so each spatial position carries location info.
  const withPos = addPosEmbed(gl, programs, embedded.output, weights.pos, SEQ_LEN);
  // 4. Stack of transformer blocks with batched self-attention.
  const txOuts = [];
  let current = withPos;
  for (const txWeights of weights.tx) {
    const txOut = batchedTransformerFwd(gl, programs, current, txWeights, SEQ_LEN, N_HEADS);
    txOuts.push(txOut);
    current = txOut.output;
  }
  // 5. Average-pool over tokens: → [batch, D_MODEL]
  const pooled = meanPool(gl, programs, current, input.rows, SEQ_LEN);
  // 6. Regression head: → [batch, OUTPUT_DIM]
  const logits = denseFwd(gl, programs, pooled, weights.head, 'linear');
  return {
    output: logits.output,
    cache: { patched, embedded, withPos, txOuts, pooled, logits },
  };
}

export function geoRegressionModelBwd(gl, programs, input, weights, cache, dOutput) {
  const gHead    = denseBwd(gl, programs, cache.pooled, weights.head, cache.logits.preActivation, dOutput, 'linear');
  let dCurrent   = meanPoolBwd(gl, programs, gHead.dInput, input.rows * SEQ_LEN, SEQ_LEN);
  destroyTensor(gl, gHead.dInput);

  const txGrads = [];
  for (let i = weights.tx.length - 1; i >= 0; i -= 1) {
    const txInput = i === 0 ? cache.withPos : cache.txOuts[i - 1].output;
    const gTx = batchedTransformerBwd(gl, programs, txInput, weights.tx[i], cache.txOuts[i].cache, dCurrent, SEQ_LEN, N_HEADS);
    txGrads.unshift(gTx.dWeights);
    destroyTensor(gl, dCurrent);
    dCurrent = gTx.dInput;
  }

  // dCurrent is the gradient w.r.t. withPos (addPosEmbed output).
  // Addition is transparent: dEmbed = dCurrent; dPos = sum over batch.
  const dPos    = posEmbGrad(gl, programs, dCurrent, SEQ_LEN);
  const gEmbed  = denseBwd(gl, programs, cache.patched, weights.embed, cache.embedded.preActivation, dCurrent, 'linear');
  destroyTensor(gl, dCurrent);
  const dInput  = unpatchify(gl, programs, gEmbed.dInput, SEQ_LEN, GRID_COLS, PATCH_SIDE, REGION_SIZE, CHANNELS);
  destroyTensor(gl, gEmbed.dInput);

  return {
    dInput,
    gradients: [
      gEmbed.dW, gEmbed.db,
      dPos,
      ...txGrads.flatMap((g) => TX_WEIGHT_KEYS.map((k) => g[k])),
      gHead.dW, gHead.db,
    ],
  };
}

export function createGeoRegressionModel(gl) {
  const weights = {
    embed: createDenseWeights(gl, TOKEN_DIM, D_MODEL),
    pos:   createTensor(gl, SEQ_LEN, D_MODEL, new Float32Array(SEQ_LEN * D_MODEL)),
    tx:    Array.from({ length: N_LAYERS }, () => createTransformerWeights(gl, D_MODEL, N_HEADS, FFN_DIM)),
    head:  createDenseWeights(gl, D_MODEL, OUTPUT_DIM),
  };
  return {
    type: 'geo-regression',
    inputSize: REGION_SIZE * REGION_SIZE * CHANNELS,
    weights,
    parameters: flattenWeights(weights),
    architecture: {
      type:       'geo-regression',
      regionSize: REGION_SIZE,
      patchSide:  PATCH_SIDE,
      seqLen:     SEQ_LEN,
      dModel:     D_MODEL,
      nHeads:     N_HEADS,
      ffnDim:     FFN_DIM,
      nLayers:    N_LAYERS,
      outputDim:  OUTPUT_DIM,
    },
    forward:  geoRegressionModelFwd,
    backward: geoRegressionModelBwd,
    disposeCache(glContext, cache) {
      destroyTensor(glContext, cache.patched);
      destroyTensor(glContext, cache.embedded.output);
      destroyTensor(glContext, cache.embedded.preActivation);
      destroyTensor(glContext, cache.withPos);
      for (const txOut of cache.txOuts) {
        destroyTensor(glContext, txOut.output);
        destroyTransformerCache(glContext, txOut.cache);
      }
      destroyTensor(glContext, cache.pooled);
      destroyTensor(glContext, cache.logits.preActivation);
      // logits.output is the model output tensor, destroyed by trainStep
    },
  };
}
