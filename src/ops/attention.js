import { createTensor, destroyTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';
import { matmul, matmulAt, matmulBt } from './matmul.js';
import { scale } from './elementwise.js';
import { softmaxFwd } from './reduce.js';

export const ATTENTION_SHADERS = {
  // Block-diagonal scores: Q,K [batch*seqLen, headDim] -> scores [batch*seqLen, seqLen].
  // Each row i only attends to the seqLen positions within its own block (i / seqLen).
  batchedAttnScores: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_Q;
uniform sampler2D u_K;
uniform int u_HeadDim;
uniform int u_SeqLen;
uniform float u_Scale;
void main() {
  int j = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  int block = i / u_SeqLen;
  float sum = 0.0;
  for (int d = 0; d < u_HeadDim; d++) {
    sum += texelFetch(u_Q, ivec2(d, i), 0).r
         * texelFetch(u_K, ivec2(d, block * u_SeqLen + j), 0).r;
  }
  outColor = sum * u_Scale;
}`,

  // Weighted V sum per block: attn [batch*seqLen, seqLen], V [batch*seqLen, headDim] -> [batch*seqLen, headDim].
  batchedAttnV: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_Attn;
uniform sampler2D u_V;
uniform int u_SeqLen;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  int block = row / u_SeqLen;
  float sum = 0.0;
  for (int t = 0; t < u_SeqLen; t++) {
    sum += texelFetch(u_Attn, ivec2(t, row), 0).r
         * texelFetch(u_V, ivec2(col, block * u_SeqLen + t), 0).r;
  }
  outColor = sum;
}`,

  // Gradient w.r.t. V: dV[block*seqLen+j, d] = sum_i attn[block*seqLen+i, j] * dOut[block*seqLen+i, d].
  batchedAttnBackV: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_Attn;
uniform sampler2D u_dOut;
uniform int u_SeqLen;
void main() {
  int d = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  int block = row / u_SeqLen;
  int j = row % u_SeqLen;
  float sum = 0.0;
  for (int i = 0; i < u_SeqLen; i++) {
    sum += texelFetch(u_Attn, ivec2(j, block * u_SeqLen + i), 0).r
         * texelFetch(u_dOut, ivec2(d, block * u_SeqLen + i), 0).r;
  }
  outColor = sum;
}`,

  // Gradient w.r.t. pre-softmax attn weights (before softmax bwd).
  batchedAttnBackAttn: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_dOut;
uniform sampler2D u_V;
uniform int u_HeadDim;
uniform int u_SeqLen;
void main() {
  int j = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  int block = i / u_SeqLen;
  float sum = 0.0;
  for (int d = 0; d < u_HeadDim; d++) {
    sum += texelFetch(u_dOut, ivec2(d, i), 0).r
         * texelFetch(u_V, ivec2(d, block * u_SeqLen + j), 0).r;
  }
  outColor = sum;
}`,

  // Gradient w.r.t. Q: dQ[i, d] = sum_j dScores[i,j] * K[block*seqLen+j, d] * scale.
  batchedAttnBackQ: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_dScores;
uniform sampler2D u_K;
uniform int u_SeqLen;
uniform float u_Scale;
void main() {
  int d = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  int block = i / u_SeqLen;
  float sum = 0.0;
  for (int j = 0; j < u_SeqLen; j++) {
    sum += texelFetch(u_dScores, ivec2(j, i), 0).r
         * texelFetch(u_K, ivec2(d, block * u_SeqLen + j), 0).r;
  }
  outColor = sum * u_Scale;
}`,

  // Gradient w.r.t. K: dK[block*seqLen+j, d] = sum_i dScores[block*seqLen+i, j] * Q[block*seqLen+i, d] * scale.
  batchedAttnBackK: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_dScores;
uniform sampler2D u_Q;
uniform int u_SeqLen;
uniform float u_Scale;
void main() {
  int d = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  int block = row / u_SeqLen;
  int j = row % u_SeqLen;
  float sum = 0.0;
  for (int i = 0; i < u_SeqLen; i++) {
    sum += texelFetch(u_dScores, ivec2(j, block * u_SeqLen + i), 0).r
         * texelFetch(u_Q, ivec2(d, block * u_SeqLen + i), 0).r;
  }
  outColor = sum * u_Scale;
}`,

  attentionSoftmaxBwd: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_Attn;
uniform sampler2D u_dAttn;
uniform int u_N;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float rowDot = 0.0;
  for (int j = 0; j < u_N; j++) {
    float a = texelFetch(u_Attn, ivec2(j, row), 0).r;
    float da = texelFetch(u_dAttn, ivec2(j, row), 0).r;
    rowDot += a * da;
  }
  float attn = texelFetch(u_Attn, ivec2(col, row), 0).r;
  float dAttn = texelFetch(u_dAttn, ivec2(col, row), 0).r;
  outColor = attn * (dAttn - rowDot);
}`,
};

export function scaledDotProductFwd(gl, programs, Q, K, V) {
  const dk = Q.cols;
  const scores = matmulBt(gl, programs, Q, K);
  const scaledScores = scale(gl, programs, scores, 1 / Math.sqrt(dk));
  const attnWeights = softmaxFwd(gl, programs, scaledScores);
  const output = matmul(gl, programs, attnWeights, V);
  destroyTensor(gl, scores);
  destroyTensor(gl, scaledScores);
  return { output, attnWeights };
}

export function scaledDotProductBwd(gl, programs, Q, K, V, attnWeights, dOut) {
  const dV = matmulAt(gl, programs, attnWeights, dOut);
  const dAttn = matmulBt(gl, programs, dOut, V);
  const dScaledScores = createTensor(gl, attnWeights.rows, attnWeights.cols);
  executePass(gl, programs.attentionSoftmaxBwd, {
    u_Attn: attnWeights.texture,
    u_dAttn: dAttn.texture,
    u_N: attnWeights.cols,
  }, dScaledScores, programs.__quadBuffer);
  destroyTensor(gl, dAttn);

  const scaleFactor = 1 / Math.sqrt(Q.cols);
  const dScores = scale(gl, programs, dScaledScores, scaleFactor);
  destroyTensor(gl, dScaledScores);

  const dQ = matmul(gl, programs, dScores, K);
  const dK = matmulAt(gl, programs, dScores, Q);
  destroyTensor(gl, dScores);

  return { dQ, dK, dV };
}

// Batched attention where attention is computed independently within each block
// of seqLen rows. Input tensors have shape [batch*seqLen, headDim].
export function batchedScaledDotProductFwd(gl, programs, Q, K, V, seqLen) {
  const headDim = Q.cols;
  const attnScale = 1 / Math.sqrt(headDim);
  const scores = createTensor(gl, Q.rows, seqLen);
  executePass(gl, programs.batchedAttnScores, {
    u_Q: Q.texture,
    u_K: K.texture,
    u_HeadDim: headDim,
    u_SeqLen: seqLen,
    u_Scale: attnScale,
  }, scores, programs.__quadBuffer);
  const attnWeights = softmaxFwd(gl, programs, scores);
  destroyTensor(gl, scores);
  const output = createTensor(gl, Q.rows, headDim);
  executePass(gl, programs.batchedAttnV, {
    u_Attn: attnWeights.texture,
    u_V: V.texture,
    u_SeqLen: seqLen,
  }, output, programs.__quadBuffer);
  return { output, attnWeights };
}

export function batchedScaledDotProductBwd(gl, programs, Q, K, V, attnWeights, dOut, seqLen) {
  const headDim = Q.cols;
  const attnScale = 1 / Math.sqrt(headDim);

  const dV = createTensor(gl, Q.rows, headDim);
  executePass(gl, programs.batchedAttnBackV, {
    u_Attn: attnWeights.texture,
    u_dOut: dOut.texture,
    u_SeqLen: seqLen,
  }, dV, programs.__quadBuffer);

  const dAttn = createTensor(gl, Q.rows, seqLen);
  executePass(gl, programs.batchedAttnBackAttn, {
    u_dOut: dOut.texture,
    u_V: V.texture,
    u_HeadDim: headDim,
    u_SeqLen: seqLen,
  }, dAttn, programs.__quadBuffer);

  const dScores = createTensor(gl, Q.rows, seqLen);
  executePass(gl, programs.attentionSoftmaxBwd, {
    u_Attn: attnWeights.texture,
    u_dAttn: dAttn.texture,
    u_N: seqLen,
  }, dScores, programs.__quadBuffer);
  destroyTensor(gl, dAttn);

  const dQ = createTensor(gl, Q.rows, headDim);
  executePass(gl, programs.batchedAttnBackQ, {
    u_dScores: dScores.texture,
    u_K: K.texture,
    u_SeqLen: seqLen,
    u_Scale: attnScale,
  }, dQ, programs.__quadBuffer);

  const dK = createTensor(gl, Q.rows, headDim);
  executePass(gl, programs.batchedAttnBackK, {
    u_dScores: dScores.texture,
    u_Q: Q.texture,
    u_SeqLen: seqLen,
    u_Scale: attnScale,
  }, dK, programs.__quadBuffer);
  destroyTensor(gl, dScores);

  return { dQ, dK, dV };
}
