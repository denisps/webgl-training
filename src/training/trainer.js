import { createTensor, destroyTensor } from '../webgl/context.js';
import { crossEntropyLoss, mseLoss } from './loss.js';
import { createAdamState, adamStep } from './optimizer.js';
import { getBatch, shuffleDataset } from '../utils/data.js';

function batchToTensor(gl, batchInputs) {
  const rows = batchInputs.length;
  const cols = batchInputs[0].length;
  const data = new Float32Array(rows * cols);
  batchInputs.forEach((sample, row) => {
    data.set(sample, row * cols);
  });
  return createTensor(gl, rows, cols, data);
}

export function createTrainer(gl, programs, model, config) {
  const shapes = model.parameters.map((tensor) => ({ rows: tensor.rows, cols: tensor.cols }));
  return {
    model,
    config,
    adamState: createAdamState(gl, shapes),
  };
}

export function trainStep(gl, programs, trainer, batchInputs, batchLabels) {
  const inputTensor = batchToTensor(gl, batchInputs);
  const { output, cache } = trainer.model.forward(gl, programs, inputTensor, trainer.model.weights);
  const lossResult = trainer.config.loss === 'mse'
    ? mseLoss(gl, programs, output, batchLabels)
    : crossEntropyLoss(gl, programs, output, batchLabels);
  const backwardResult = trainer.model.backward(gl, programs, inputTensor, trainer.model.weights, cache, lossResult.dLogits ?? lossResult.dPredictions);
  adamStep(gl, programs, trainer.model.parameters, backwardResult.gradients, trainer.adamState, trainer.config);

  destroyTensor(gl, inputTensor);
  destroyTensor(gl, output);
  destroyTensor(gl, lossResult.dLogits ?? lossResult.dPredictions);
  destroyTensor(gl, backwardResult.dInput);
  trainer.model.disposeCache(gl, cache);
  backwardResult.gradients.forEach((tensor) => destroyTensor(gl, tensor));

  return { loss: lossResult.loss };
}

export function trainEpoch(gl, programs, trainer, dataset, onBatch) {
  const shuffled = shuffleDataset(dataset);
  const batchSize = trainer.config.batchSize;
  const batchCount = Math.ceil(shuffled.inputs.length / batchSize);
  let totalLoss = 0;
  for (let batchIndex = 0; batchIndex < batchCount; batchIndex += 1) {
    const batch = getBatch(shuffled, batchIndex, batchSize);
    const result = trainStep(gl, programs, trainer, batch.inputs, batch.labels);
    totalLoss += result.loss;
    if (onBatch) {
      onBatch({ batchIndex, batchCount, loss: result.loss });
    }
  }
  return { avgLoss: totalLoss / Math.max(batchCount, 1) };
}
