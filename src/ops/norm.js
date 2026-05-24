import { createTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
`;

export const NORM_SHADERS = {
  layerNormMean: `${HEADER}
uniform sampler2D u_X;
uniform int u_N;
void main() {
  int row = int(gl_FragCoord.y);
  float mean = 0.0;
  for (int col = 0; col < u_N; col++) {
    mean += texelFetch(u_X, ivec2(col, row), 0).r;
  }
  outColor = mean / float(u_N);
}`,
  layerNormInvStd: `${HEADER}
uniform sampler2D u_X;
uniform sampler2D u_Mean;
uniform int u_N;
uniform float u_Eps;
void main() {
  int row = int(gl_FragCoord.y);
  float mean = texelFetch(u_Mean, ivec2(0, row), 0).r;
  float variance = 0.0;
  for (int col = 0; col < u_N; col++) {
    float diff = texelFetch(u_X, ivec2(col, row), 0).r - mean;
    variance += diff * diff;
  }
  outColor = inversesqrt(variance / float(u_N) + u_Eps);
}`,
  layerNormApply: `${HEADER}
uniform sampler2D u_X;
uniform sampler2D u_Gamma;
uniform sampler2D u_Beta;
uniform sampler2D u_Mean;
uniform sampler2D u_InvStd;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float mean = texelFetch(u_Mean, ivec2(0, row), 0).r;
  float invStd = texelFetch(u_InvStd, ivec2(0, row), 0).r;
  float x = texelFetch(u_X, ivec2(col, row), 0).r;
  float gamma = texelFetch(u_Gamma, ivec2(col, 0), 0).r;
  float beta = texelFetch(u_Beta, ivec2(col, 0), 0).r;
  outColor = gamma * ((x - mean) * invStd) + beta;
}`,
  layerNormDGamma: `${HEADER}
uniform sampler2D u_X;
uniform sampler2D u_Mean;
uniform sampler2D u_InvStd;
uniform sampler2D u_dY;
uniform int u_M;
void main() {
  int col = int(gl_FragCoord.x);
  float sumValue = 0.0;
  for (int row = 0; row < u_M; row++) {
    float x = texelFetch(u_X, ivec2(col, row), 0).r;
    float mean = texelFetch(u_Mean, ivec2(0, row), 0).r;
    float invStd = texelFetch(u_InvStd, ivec2(0, row), 0).r;
    float dy = texelFetch(u_dY, ivec2(col, row), 0).r;
    sumValue += dy * ((x - mean) * invStd);
  }
  outColor = sumValue;
}`,
  layerNormDBeta: `${HEADER}
uniform sampler2D u_dY;
uniform int u_M;
void main() {
  int col = int(gl_FragCoord.x);
  float sumValue = 0.0;
  for (int row = 0; row < u_M; row++) {
    sumValue += texelFetch(u_dY, ivec2(col, row), 0).r;
  }
  outColor = sumValue;
}`,
  layerNormDX: `${HEADER}
uniform sampler2D u_X;
uniform sampler2D u_Gamma;
uniform sampler2D u_Mean;
uniform sampler2D u_InvStd;
uniform sampler2D u_dY;
uniform int u_N;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float mean = texelFetch(u_Mean, ivec2(0, row), 0).r;
  float invStd = texelFetch(u_InvStd, ivec2(0, row), 0).r;
  float x = texelFetch(u_X, ivec2(col, row), 0).r;
  float xhat = (x - mean) * invStd;
  float sumDyGamma = 0.0;
  float sumDyGammaXhat = 0.0;
  for (int c = 0; c < u_N; c++) {
    float gamma = texelFetch(u_Gamma, ivec2(c, 0), 0).r;
    float dy = texelFetch(u_dY, ivec2(c, row), 0).r;
    float xc = texelFetch(u_X, ivec2(c, row), 0).r;
    float xhatC = (xc - mean) * invStd;
    float dyGamma = dy * gamma;
    sumDyGamma += dyGamma;
    sumDyGammaXhat += dyGamma * xhatC;
  }
  float gamma = texelFetch(u_Gamma, ivec2(col, 0), 0).r;
  float dy = texelFetch(u_dY, ivec2(col, row), 0).r;
  float dyGamma = dy * gamma;
  outColor = invStd / float(u_N) * (float(u_N) * dyGamma - sumDyGamma - xhat * sumDyGammaXhat);
}`,
};

export function layerNormFwd(gl, programs, X, gamma, beta, eps = 1e-5) {
  const mean = createTensor(gl, X.rows, 1);
  const invStd = createTensor(gl, X.rows, 1);
  const output = createTensor(gl, X.rows, X.cols);

  executePass(gl, programs.layerNormMean, {
    u_X: X.texture,
    u_N: X.cols,
  }, mean, programs.__quadBuffer);

  executePass(gl, programs.layerNormInvStd, {
    u_X: X.texture,
    u_Mean: mean.texture,
    u_N: X.cols,
    u_Eps: eps,
  }, invStd, programs.__quadBuffer);

  executePass(gl, programs.layerNormApply, {
    u_X: X.texture,
    u_Gamma: gamma.texture,
    u_Beta: beta.texture,
    u_Mean: mean.texture,
    u_InvStd: invStd.texture,
  }, output, programs.__quadBuffer);

  return { output, mean, invStd };
}

export function layerNormBwd(gl, programs, X, gamma, mean, invStd, dY) {
  const dGamma = createTensor(gl, 1, X.cols);
  const dBeta = createTensor(gl, 1, X.cols);
  const dX = createTensor(gl, X.rows, X.cols);

  executePass(gl, programs.layerNormDGamma, {
    u_X: X.texture,
    u_Mean: mean.texture,
    u_InvStd: invStd.texture,
    u_dY: dY.texture,
    u_M: X.rows,
  }, dGamma, programs.__quadBuffer);

  executePass(gl, programs.layerNormDBeta, {
    u_dY: dY.texture,
    u_M: X.rows,
  }, dBeta, programs.__quadBuffer);

  executePass(gl, programs.layerNormDX, {
    u_X: X.texture,
    u_Gamma: gamma.texture,
    u_Mean: mean.texture,
    u_InvStd: invStd.texture,
    u_dY: dY.texture,
    u_N: X.cols,
  }, dX, programs.__quadBuffer);

  return { dX, dGamma, dBeta };
}
