import { createTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_X;
`;

export const REDUCE_SHADERS = {
  softmaxFwd: `${HEADER}
uniform int u_N;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float maxValue = texelFetch(u_X, ivec2(0, row), 0).r;
  for (int n = 1; n < u_N; n++) {
    maxValue = max(maxValue, texelFetch(u_X, ivec2(n, row), 0).r);
  }
  float sumExp = 0.0;
  for (int n = 0; n < u_N; n++) {
    sumExp += exp(texelFetch(u_X, ivec2(n, row), 0).r - maxValue);
  }
  float value = texelFetch(u_X, ivec2(col, row), 0).r;
  outColor = exp(value - maxValue) / sumExp;
}`,
  logSoftmaxFwd: `${HEADER}
uniform int u_N;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float maxValue = texelFetch(u_X, ivec2(0, row), 0).r;
  for (int n = 1; n < u_N; n++) {
    maxValue = max(maxValue, texelFetch(u_X, ivec2(n, row), 0).r);
  }
  float sumExp = 0.0;
  for (int n = 0; n < u_N; n++) {
    sumExp += exp(texelFetch(u_X, ivec2(n, row), 0).r - maxValue);
  }
  float logSum = log(sumExp) + maxValue;
  float value = texelFetch(u_X, ivec2(col, row), 0).r;
  outColor = value - logSum;
}`,
  sumCols: `${HEADER}
uniform int u_M;
void main() {
  int col = int(gl_FragCoord.x);
  float sumValue = 0.0;
  for (int row = 0; row < u_M; row++) {
    sumValue += texelFetch(u_X, ivec2(col, row), 0).r;
  }
  outColor = sumValue;
}`,
  l2NormFwd: `${HEADER}
uniform int u_N;
uniform float u_Eps;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float sumSq = 0.0;
  for (int n = 0; n < u_N; n++) {
    float v = texelFetch(u_X, ivec2(n, row), 0).r;
    sumSq += v * v;
  }
  outColor = texelFetch(u_X, ivec2(col, row), 0).r / (sqrt(sumSq) + u_Eps);
}`,
};

export function softmaxFwd(gl, programs, X) {
  const output = createTensor(gl, X.rows, X.cols);
  executePass(gl, programs.softmaxFwd, {
    u_X: X.texture,
    u_N: X.cols,
  }, output, programs.__quadBuffer);
  return output;
}

export function logSoftmaxFwd(gl, programs, X) {
  const output = createTensor(gl, X.rows, X.cols);
  executePass(gl, programs.logSoftmaxFwd, {
    u_X: X.texture,
    u_N: X.cols,
  }, output, programs.__quadBuffer);
  return output;
}

export function sumCols(gl, programs, A) {
  const output = createTensor(gl, 1, A.cols);
  executePass(gl, programs.sumCols, {
    u_X: A.texture,
    u_M: A.rows,
  }, output, programs.__quadBuffer);
  return output;
}

export function l2NormFwd(gl, programs, X, eps = 1e-8) {
  const output = createTensor(gl, X.rows, X.cols);
  executePass(gl, programs.l2NormFwd, {
    u_X: X.texture,
    u_N: X.cols,
    u_Eps: eps,
  }, output, programs.__quadBuffer);
  return output;
}
