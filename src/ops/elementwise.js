import { createTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_A;
`;

const BINARY_HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_A;
uniform sampler2D u_B;
`;

export const ELEMENTWISE_SHADERS = {
  reluFwd: `${HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  outColor = max(0.0, x);
}`,
  reluBwd: `${BINARY_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  float dy = texelFetch(u_B, ivec2(col, row), 0).r;
  outColor = x > 0.0 ? dy : 0.0;
}`,
  geluFwd: `${HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  float c = 0.7978845608028654;
  float inner = c * (x + 0.044715 * x * x * x);
  outColor = 0.5 * x * (1.0 + tanh(inner));
}`,
  geluBwd: `${BINARY_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  float dy = texelFetch(u_B, ivec2(col, row), 0).r;
  float c = 0.7978845608028654;
  float x2 = x * x;
  float inner = c * (x + 0.044715 * x * x2);
  float t = tanh(inner);
  float sech2 = 1.0 - t * t;
  float derivative = 0.5 * (1.0 + t) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x2);
  outColor = derivative * dy;
}`,
  sigmoidFwd: `${HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  outColor = 1.0 / (1.0 + exp(-x));
}`,
  sigmoidBwd: `${BINARY_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  float dy = texelFetch(u_B, ivec2(col, row), 0).r;
  float s = 1.0 / (1.0 + exp(-x));
  outColor = dy * s * (1.0 - s);
}`,
  add: `${BINARY_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  outColor = texelFetch(u_A, ivec2(col, row), 0).r + texelFetch(u_B, ivec2(col, row), 0).r;
}`,
  mul: `${BINARY_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  outColor = texelFetch(u_A, ivec2(col, row), 0).r * texelFetch(u_B, ivec2(col, row), 0).r;
}`,
  scale: `${HEADER}
uniform float u_Scale;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  outColor = texelFetch(u_A, ivec2(col, row), 0).r * u_Scale;
}`,
  addBias: `${BINARY_HEADER}
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float a = texelFetch(u_A, ivec2(col, row), 0).r;
  float bias = texelFetch(u_B, ivec2(col, 0), 0).r;
  outColor = a + bias;
}`,
  logFwd: `${HEADER}
uniform float u_Eps;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float x = texelFetch(u_A, ivec2(col, row), 0).r;
  outColor = log(max(x, u_Eps));
}`,
};

function unary(gl, programs, name, A, extraUniforms = {}) {
  const output = createTensor(gl, A.rows, A.cols);
  executePass(gl, programs[name], { u_A: A.texture, ...extraUniforms }, output, programs.__quadBuffer);
  return output;
}

function binary(gl, programs, name, A, B, extraUniforms = {}) {
  const output = createTensor(gl, A.rows, A.cols);
  executePass(gl, programs[name], { u_A: A.texture, u_B: B.texture, ...extraUniforms }, output, programs.__quadBuffer);
  return output;
}

export function reluFwd(gl, programs, X) {
  return unary(gl, programs, 'reluFwd', X);
}

export function reluBwd(gl, programs, X, dY) {
  return binary(gl, programs, 'reluBwd', X, dY);
}

export function geluFwd(gl, programs, X) {
  return unary(gl, programs, 'geluFwd', X);
}

export function geluBwd(gl, programs, X, dY) {
  return binary(gl, programs, 'geluBwd', X, dY);
}

export function sigmoidFwd(gl, programs, X) {
  return unary(gl, programs, 'sigmoidFwd', X);
}

export function sigmoidBwd(gl, programs, X, dY) {
  return binary(gl, programs, 'sigmoidBwd', X, dY);
}

export function add(gl, programs, A, B) {
  return binary(gl, programs, 'add', A, B);
}

export function mul(gl, programs, A, B) {
  return binary(gl, programs, 'mul', A, B);
}

export function scale(gl, programs, A, s) {
  return unary(gl, programs, 'scale', A, { u_Scale: s });
}

export function addBias(gl, programs, A, bias) {
  const output = createTensor(gl, A.rows, A.cols);
  executePass(gl, programs.addBias, {
    u_A: A.texture,
    u_B: bias.texture,
  }, output, programs.__quadBuffer);
  return output;
}

export function logFwd(gl, programs, X, eps = 1e-10) {
  return unary(gl, programs, 'logFwd', X, { u_Eps: eps });
}
