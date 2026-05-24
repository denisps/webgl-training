import { createTensor, destroyTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';
import { matmul, matmulAt, matmulBt } from './matmul.js';
import { scale } from './elementwise.js';
import { softmaxFwd } from './reduce.js';

export const ATTENTION_SHADERS = {
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
