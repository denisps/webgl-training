export async function loadImageToCanvas(imageUrl) {
  const image = new Image();
  image.crossOrigin = 'anonymous';
  const canvas = document.createElement('canvas');
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = imageUrl;
  });
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  return canvas;
}

export function canvasToFloat32(canvas, width, height, normalize = true) {
  const workCanvas = document.createElement('canvas');
  workCanvas.width = width;
  workCanvas.height = height;
  const ctx = workCanvas.getContext('2d');
  ctx.drawImage(canvas, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const result = new Float32Array(width * height * 3);
  for (let index = 0; index < width * height; index += 1) {
    const source = index * 4;
    const target = index * 3;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = pixels[source + channel];
      result[target + channel] = normalize ? value / 127.5 - 1 : value;
    }
  }
  return result;
}

export function cropImage(imageData, x, y, width, height, srcWidth) {
  const srcHeight = Math.ceil(imageData.length / (srcWidth * 4));
  const channels = imageData.length / (srcWidth * srcHeight);
  const resolvedChannels = (channels >= 1 && Number.isFinite(channels)) ? Math.round(channels) : 4;
  const output = new Float32Array(width * height * resolvedChannels);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const srcIndex = ((y + row) * srcWidth + (x + col)) * resolvedChannels;
      const dstIndex = (row * width + col) * resolvedChannels;
      for (let channel = 0; channel < resolvedChannels; channel += 1) {
        output[dstIndex + channel] = imageData[srcIndex + channel];
      }
    }
  }
  return output;
}

export function resizeImageData(imageData, srcW, srcH, dstW, dstH) {
  const channels = imageData.length / (srcW * srcH);
  const output = new Float32Array(dstW * dstH * channels);
  for (let y = 0; y < dstH; y += 1) {
    const gy = ((y + 0.5) * srcH) / dstH - 0.5;
    const y0 = Math.max(Math.floor(gy), 0);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const wy = gy - y0;
    for (let x = 0; x < dstW; x += 1) {
      const gx = ((x + 0.5) * srcW) / dstW - 0.5;
      const x0 = Math.max(Math.floor(gx), 0);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const wx = gx - x0;
      for (let channel = 0; channel < channels; channel += 1) {
        const c00 = imageData[(y0 * srcW + x0) * channels + channel];
        const c10 = imageData[(y0 * srcW + x1) * channels + channel];
        const c01 = imageData[(y1 * srcW + x0) * channels + channel];
        const c11 = imageData[(y1 * srcW + x1) * channels + channel];
        const top = c00 * (1 - wx) + c10 * wx;
        const bottom = c01 * (1 - wx) + c11 * wx;
        output[(y * dstW + x) * channels + channel] = top * (1 - wy) + bottom * wy;
      }
    }
  }
  return output;
}

export function augmentImage(imageData, width, height) {
  const channels = imageData.length / (width * height);
  const output = new Float32Array(imageData.length);
  const flip = Math.random() < 0.5;
  const brightness = 1 + (Math.random() * 0.2 - 0.1);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const srcX = flip ? width - 1 - x : x;
      const srcIndex = (y * width + srcX) * channels;
      const dstIndex = (y * width + x) * channels;
      for (let channel = 0; channel < channels; channel += 1) {
        output[dstIndex + channel] = imageData[srcIndex + channel] * brightness;
      }
    }
  }
  return output;
}
