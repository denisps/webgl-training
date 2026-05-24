export function createDataset(inputs, labels) {
  return {
    inputs: [...inputs],
    labels: [...labels],
    size: labels.length,
  };
}

export function shuffleDataset(dataset) {
  const indices = [...dataset.labels.keys()];
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
  }
  return {
    inputs: indices.map((index) => dataset.inputs[index]),
    labels: indices.map((index) => dataset.labels[index]),
    size: dataset.size,
  };
}

export function getBatch(dataset, batchIndex, batchSize) {
  const start = batchIndex * batchSize;
  const end = Math.min(start + batchSize, dataset.inputs.length);
  return {
    inputs: dataset.inputs.slice(start, end),
    labels: new Int32Array(dataset.labels.slice(start, end)),
  };
}

export function splitDataset(dataset, trainFraction) {
  const splitIndex = Math.max(1, Math.min(dataset.inputs.length - 1, Math.floor(dataset.inputs.length * trainFraction)));
  return {
    train: createDataset(dataset.inputs.slice(0, splitIndex), dataset.labels.slice(0, splitIndex)),
    val: createDataset(dataset.inputs.slice(splitIndex), dataset.labels.slice(splitIndex)),
  };
}
