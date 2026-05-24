import { createTensor, destroyTensor, readTensor } from '../webgl/context.js';
import { executePass } from '../webgl/program.js';
import { softmaxFwd } from '../ops/reduce.js';

export const LOSS_SHADERS = {
  crossEntropyGrad: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_Probs;
uniform sampler2D u_Labels;
uniform int u_Batch;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float prob = texelFetch(u_Probs, ivec2(col, row), 0).r;
  int label = int(texelFetch(u_Labels, ivec2(0, row), 0).r + 0.5);
  outColor = (prob - float(col == label)) / float(u_Batch);
}`,
  mseGrad: `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out float outColor;
uniform sampler2D u_Pred;
uniform sampler2D u_Target;
uniform float u_Scale;
void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  float pred = texelFetch(u_Pred, ivec2(col, row), 0).r;
  float target = texelFetch(u_Target, ivec2(col, row), 0).r;
  outColor = (pred - target) * u_Scale;
}`,
};

export function crossEntropyLoss(gl, programs, logits, labels) {
  const probs = softmaxFwd(gl, programs, logits);
  const probsData = readTensor(gl, probs);
  let loss = 0;
  for (let row = 0; row < logits.rows; row += 1) {
    const label = labels[row];
    loss -= Math.log(Math.max(probsData[row * logits.cols + label], 1e-8));
  }
  loss /= logits.rows;

  const labelData = new Float32Array(logits.rows);
  for (let index = 0; index < labels.length; index += 1) {
    labelData[index] = labels[index];
  }
  const labelTensor = createTensor(gl, logits.rows, 1, labelData);
  const dLogits = createTensor(gl, logits.rows, logits.cols);
  executePass(gl, programs.crossEntropyGrad, {
    u_Probs: probs.texture,
    u_Labels: labelTensor.texture,
    u_Batch: logits.rows,
  }, dLogits, programs.__quadBuffer);

  destroyTensor(gl, probs);
  destroyTensor(gl, labelTensor);
  return { loss, dLogits };
}

export function mseLoss(gl, programs, predictions, targets) {
  const targetTensor = targets.texture ? targets : createTensor(gl, predictions.rows, predictions.cols, targets);
  const predData = readTensor(gl, predictions);
  const targetData = readTensor(gl, targetTensor);
  let loss = 0;
  for (let index = 0; index < predData.length; index += 1) {
    const diff = predData[index] - targetData[index];
    loss += diff * diff;
  }
  loss /= predData.length;

  const dPredictions = createTensor(gl, predictions.rows, predictions.cols);
  executePass(gl, programs.mseGrad, {
    u_Pred: predictions.texture,
    u_Target: targetTensor.texture,
    u_Scale: 2 / predData.length,
  }, dPredictions, programs.__quadBuffer);

  if (!targets.texture) {
    destroyTensor(gl, targetTensor);
  }
  return { loss, dPredictions };
}
