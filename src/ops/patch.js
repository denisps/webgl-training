import { createTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_X;`;

export const PATCH_SHADERS = {
  // Reshape [batch, seqLen*tokenDim] -> [batch*seqLen, tokenDim].
  // Patches are non-overlapping spatial tiles of a row-major image
  // stored as [imageHeight*imageWidth*channels] per row.
  patchify: `${HEADER}
uniform int u_SeqLen;
uniform int u_GridCols;
uniform int u_PatchSide;
uniform int u_ImageCols;
uniform int u_Channels;
void main() {
  int d = int(gl_FragCoord.x);
  int rowIdx = int(gl_FragCoord.y);
  int b = rowIdx / u_SeqLen;
  int t = rowIdx % u_SeqLen;
  int pi = t / u_GridCols;
  int pj = t % u_GridCols;
  int pixelIdx = d / u_Channels;
  int c = d % u_Channels;
  int local_py = pixelIdx / u_PatchSide;
  int local_px = pixelIdx % u_PatchSide;
  int src_col = (pi * u_PatchSide + local_py) * u_ImageCols * u_Channels
              + (pj * u_PatchSide + local_px) * u_Channels + c;
  outColor = texelFetch(u_X, ivec2(src_col, b), 0).r;
}`,

  // Inverse of patchify: [batch*seqLen, tokenDim] -> [batch, seqLen*tokenDim].
  unpatchify: `${HEADER}
uniform int u_SeqLen;
uniform int u_GridCols;
uniform int u_PatchSide;
uniform int u_ImageCols;
uniform int u_Channels;
void main() {
  int src_col = int(gl_FragCoord.x);
  int b = int(gl_FragCoord.y);
  int c = src_col % u_Channels;
  int totalPixels = src_col / u_Channels;
  int src_py = totalPixels / u_ImageCols;
  int src_px = totalPixels % u_ImageCols;
  int pi = src_py / u_PatchSide;
  int pj = src_px / u_PatchSide;
  int local_py = src_py % u_PatchSide;
  int local_px = src_px % u_PatchSide;
  int t = pi * u_GridCols + pj;
  int d = (local_py * u_PatchSide + local_px) * u_Channels + c;
  outColor = texelFetch(u_X, ivec2(d, b * u_SeqLen + t), 0).r;
}`,

  // Average pool tokens: [batch*seqLen, dModel] -> [batch, dModel].
  meanPool: `${HEADER}
uniform int u_SeqLen;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float sum = 0.0;
  for (int t = 0; t < u_SeqLen; t++) {
    sum += texelFetch(u_X, ivec2(col, row * u_SeqLen + t), 0).r;
  }
  outColor = sum / float(u_SeqLen);
}`,

  // Backward of meanPool: broadcast [batch, dModel] -> [batch*seqLen, dModel].
  meanPoolBwd: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_dOut;
uniform int u_SeqLen;
void main() {
  int col = int(gl_FragCoord.x);
  int rowIdx = int(gl_FragCoord.y);
  int b = rowIdx / u_SeqLen;
  outColor = texelFetch(u_dOut, ivec2(col, b), 0).r / float(u_SeqLen);
}`,
};

export function patchify(gl, programs, input, seqLen, gridCols, patchSide, imageWidth, channels) {
  const tokenDim = patchSide * patchSide * channels;
  const output = createTensor(gl, input.rows * seqLen, tokenDim);
  executePass(gl, programs.patchify, {
    u_X: input.texture,
    u_SeqLen: seqLen,
    u_GridCols: gridCols,
    u_PatchSide: patchSide,
    u_ImageCols: imageWidth,
    u_Channels: channels,
  }, output, programs.__quadBuffer);
  return output;
}

export function unpatchify(gl, programs, input, seqLen, gridCols, patchSide, imageWidth, channels) {
  const batch = input.rows / seqLen;
  const totalDim = input.cols * seqLen;
  const output = createTensor(gl, batch, totalDim);
  executePass(gl, programs.unpatchify, {
    u_X: input.texture,
    u_SeqLen: seqLen,
    u_GridCols: gridCols,
    u_PatchSide: patchSide,
    u_ImageCols: imageWidth,
    u_Channels: channels,
  }, output, programs.__quadBuffer);
  return output;
}

export function meanPool(gl, programs, input, batch, seqLen) {
  const output = createTensor(gl, batch, input.cols);
  executePass(gl, programs.meanPool, {
    u_X: input.texture,
    u_SeqLen: seqLen,
  }, output, programs.__quadBuffer);
  return output;
}

export function meanPoolBwd(gl, programs, dOut, totalRows, seqLen) {
  const output = createTensor(gl, totalRows, dOut.cols);
  executePass(gl, programs.meanPoolBwd, {
    u_dOut: dOut.texture,
    u_SeqLen: seqLen,
  }, output, programs.__quadBuffer);
  return output;
}
