import { destroyTensor } from '../../src/webgl/context.js';
import { denseBwd, denseFwd, createDenseWeights } from '../../src/layers/dense.js';
import { createTransformerWeights, batchedTransformerFwd, batchedTransformerBwd, destroyTransformerCache } from '../../src/layers/transformer.js';
import { patchify, unpatchify, meanPool, meanPoolBwd } from '../../src/ops/patch.js';

// ViT-style architecture: split each REGION_SIZE×REGION_SIZE×3 patch into
// (REGION_SIZE/PATCH_SIDE)² spatial tokens of PATCH_SIDE²×3 features each,
// project to D_MODEL, run a transformer block, then average-pool before the head.
// Change REGION_SIZE (must be a multiple of PATCH_SIDE) to use larger input crops.
export const REGION_SIZE = 64;
const PATCH_SIDE = 8;
const CHANNELS = 3;
const SEQ_LEN = (REGION_SIZE / PATCH_SIDE) ** 2;   // 64 tokens for 64×64
const GRID_COLS = REGION_SIZE / PATCH_SIDE;          // 8 for 64×64
const TOKEN_DIM = PATCH_SIDE * PATCH_SIDE * CHANNELS; // 192
const D_MODEL = 128;
const N_HEADS = 4;
const FFN_DIM = 256;

function flattenWeights(weights) {
  return [
    weights.embed.W, weights.embed.b,
    weights.tx.Wq, weights.tx.Wk, weights.tx.Wv, weights.tx.Wo,
    weights.tx.W1, weights.tx.b1, weights.tx.W2, weights.tx.b2,
    weights.tx.gamma1, weights.tx.beta1, weights.tx.gamma2, weights.tx.beta2,
    weights.head.W, weights.head.b,
  ];
}

export function mapModelFwd(gl, programs, input, weights) {
  // 1. Split each sample into spatial tokens: [batch, inputSize] -> [batch*SEQ_LEN, TOKEN_DIM]
  const patched = patchify(gl, programs, input, SEQ_LEN, GRID_COLS, PATCH_SIDE, REGION_SIZE, CHANNELS);
  // 2. Shared linear projection for all tokens: -> [batch*SEQ_LEN, D_MODEL]
  const embedded = denseFwd(gl, programs, patched, weights.embed, 'linear');
  // 3. Transformer block with per-sample block-diagonal attention: -> [batch*16, 128]
  const txOut = batchedTransformerFwd(gl, programs, embedded.output, weights.tx, SEQ_LEN, N_HEADS);
  // 4. Average-pool the 16 token representations per sample: -> [batch, 128]
  const pooled = meanPool(gl, programs, txOut.output, input.rows, SEQ_LEN);
  // 5. Classification head: -> [batch, nRegions]
  const logits = denseFwd(gl, programs, pooled, weights.head, 'linear');
  return {
    output: logits.output,
    cache: { patched, embedded, txOut, pooled, logits },
  };
}

export function mapModelBwd(gl, programs, input, weights, cache, dOutput) {
  const gHead = denseBwd(gl, programs, cache.pooled, weights.head, cache.logits.preActivation, dOutput, 'linear');
  const dTxOut = meanPoolBwd(gl, programs, gHead.dInput, cache.txOut.output.rows, SEQ_LEN);
  destroyTensor(gl, gHead.dInput);
  const gTx = batchedTransformerBwd(gl, programs, cache.embedded.output, weights.tx, cache.txOut.cache, dTxOut, SEQ_LEN, N_HEADS);
  destroyTensor(gl, dTxOut);
  const gEmbed = denseBwd(gl, programs, cache.patched, weights.embed, cache.embedded.preActivation, gTx.dInput, 'linear');
  destroyTensor(gl, gTx.dInput);
  const dInput = unpatchify(gl, programs, gEmbed.dInput, SEQ_LEN, GRID_COLS, PATCH_SIDE, REGION_SIZE, CHANNELS);
  destroyTensor(gl, gEmbed.dInput);
  return {
    dInput,
    gradients: [
      gEmbed.dW, gEmbed.db,
      gTx.dWeights.Wq, gTx.dWeights.Wk, gTx.dWeights.Wv, gTx.dWeights.Wo,
      gTx.dWeights.W1, gTx.dWeights.b1, gTx.dWeights.W2, gTx.dWeights.b2,
      gTx.dWeights.gamma1, gTx.dWeights.beta1, gTx.dWeights.gamma2, gTx.dWeights.beta2,
      gHead.dW, gHead.db,
    ],
  };
}

export function createMapModel(gl, nRegions = 8) {
  const weights = {
    embed: createDenseWeights(gl, TOKEN_DIM, D_MODEL),
    tx: createTransformerWeights(gl, D_MODEL, N_HEADS, FFN_DIM),
    head: createDenseWeights(gl, D_MODEL, nRegions),
  };
  return {
    type: 'patch-transformer',
    inputSize: REGION_SIZE * REGION_SIZE * CHANNELS,
    weights,
    parameters: flattenWeights(weights),
    architecture: {
      type: 'patch-transformer',
      nRegions,
      patchSide: PATCH_SIDE,
      seqLen: SEQ_LEN,
      dModel: D_MODEL,
      nHeads: N_HEADS,
      ffnDim: FFN_DIM,
    },
    forward: mapModelFwd,
    backward: mapModelBwd,
    disposeCache(glContext, cache) {
      destroyTensor(glContext, cache.patched);
      destroyTensor(glContext, cache.embedded.output);
      destroyTensor(glContext, cache.embedded.preActivation);
      destroyTensor(glContext, cache.txOut.output);
      destroyTransformerCache(glContext, cache.txOut.cache);
      destroyTensor(glContext, cache.pooled);
      destroyTensor(glContext, cache.logits.preActivation);
      // logits.output is the model output tensor, destroyed by trainStep
    },
  };
}
