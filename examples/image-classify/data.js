function addNoise(value, amount) {
  return Math.max(-1, Math.min(1, value + (Math.random() * 2 - 1) * amount));
}

function buildPattern(classIndex, size) {
  const image = new Float32Array(size * size);
  const center = (size - 1) / 2;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      let value = -1;
      const diagonal = Math.abs(x - y) < 2;
      const antiDiagonal = Math.abs(x + y - (size - 1)) < 2;
      const stripeX = x % 6 < 3;
      const stripeY = y % 6 < 3;
      const dist = Math.hypot(x - center, y - center);
      switch (classIndex % 6) {
        case 0:
          value = stripeX ? 1 : -1;
          break;
        case 1:
          value = stripeY ? 1 : -1;
          break;
        case 2:
          value = diagonal ? 1 : -1;
          break;
        case 3:
          value = antiDiagonal ? 1 : -1;
          break;
        case 4:
          value = dist > size * 0.22 && dist < size * 0.34 ? 1 : -1;
          break;
        default:
          value = stripeX === stripeY ? 1 : -1;
          break;
      }
      image[y * size + x] = value;
    }
  }
  return image;
}

export function generatePatternDataset(nClasses, samplesPerClass) {
  const inputs = [];
  const labels = [];
  const size = 28;
  for (let label = 0; label < nClasses; label += 1) {
    const base = buildPattern(label, size);
    for (let sample = 0; sample < samplesPerClass; sample += 1) {
      const input = new Float32Array(base.length);
      for (let index = 0; index < base.length; index += 1) {
        input[index] = addNoise(base[index], 0.15);
      }
      inputs.push(input);
      labels.push(label);
    }
  }
  return { inputs, labels };
}
