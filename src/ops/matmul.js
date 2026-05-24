import { createTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';

const SHADER_HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_A;
uniform sampler2D u_B;
uniform int u_K;
`;

export const MATMUL_SHADERS = {
  matmul: `${SHADER_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float sum = 0.0;
  for (int k = 0; k < u_K; k++) {
    float a = texelFetch(u_A, ivec2(k, row), 0).r;
    float b = texelFetch(u_B, ivec2(col, k), 0).r;
    sum += a * b;
  }
  outColor = sum;
}`,
  matmulAt: `${SHADER_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float sum = 0.0;
  for (int k = 0; k < u_K; k++) {
    float a = texelFetch(u_A, ivec2(row, k), 0).r;
    float b = texelFetch(u_B, ivec2(col, k), 0).r;
    sum += a * b;
  }
  outColor = sum;
}`,
  matmulBt: `${SHADER_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float sum = 0.0;
  for (int k = 0; k < u_K; k++) {
    float a = texelFetch(u_A, ivec2(k, row), 0).r;
    float b = texelFetch(u_B, ivec2(k, col), 0).r;
    sum += a * b;
  }
  outColor = sum;
}`,
};

export function matmul(gl, programs, A, B) {
  const output = createTensor(gl, A.rows, B.cols);
  executePass(gl, programs.matmul, {
    u_A: A.texture,
    u_B: B.texture,
    u_K: A.cols,
  }, output, programs.__quadBuffer);
  return output;
}

export function matmulAt(gl, programs, A, B) {
  const output = createTensor(gl, A.cols, B.cols);
  executePass(gl, programs.matmulAt, {
    u_A: A.texture,
    u_B: B.texture,
    u_K: A.rows,
  }, output, programs.__quadBuffer);
  return output;
}

export function matmulBt(gl, programs, A, B) {
  const output = createTensor(gl, A.rows, B.rows);
  executePass(gl, programs.matmulBt, {
    u_A: A.texture,
    u_B: B.texture,
    u_K: A.cols,
  }, output, programs.__quadBuffer);
  return output;
}
