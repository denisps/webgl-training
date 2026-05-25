// Training scale factor range in log2 space (scaleFactor = source pixels per patch pixel).
// Range [2^-1, 2^2] = [0.5, 4] covers fine-detail through wider context crops.
export const TRAIN_LOG2_SCALE_MIN = -1;
export const TRAIN_LOG2_SCALE_MAX = 2;

// Encode scaleFactor → [0, 1] within the training range.
export function encodeScaleNorm(scaleFactor) {
  const log2Scale = Math.log2(Math.max(1e-9, scaleFactor));
  return (log2Scale - TRAIN_LOG2_SCALE_MIN) / (TRAIN_LOG2_SCALE_MAX - TRAIN_LOG2_SCALE_MIN);
}

// Decode [0, 1] scale label back to scaleFactor.
export function decodeScaleNorm(scaleNorm) {
  return Math.pow(2, scaleNorm * (TRAIN_LOG2_SCALE_MAX - TRAIN_LOG2_SCALE_MIN) + TRAIN_LOG2_SCALE_MIN);
}

/**
 * Extract a patchSize×patchSize RGB patch from `source` (ImageBitmap, canvas, or video)
 * centered at (cx, cy) in source coordinates. `scaleFactor` source pixels map to one patch
 * pixel (>1 = zoomed out). `rotationRad` is a clockwise pre-rotation applied before sampling,
 * e.g. -compassHeading to align north-up.
 *
 * The canvas transform chain is: translate to patch centre → rotate → scale → translate to (cx,cy).
 * Areas that fall outside the source image are filled with black (canvas default).
 *
 * Returns a Float32Array of length patchSize²×3, values normalised to [-1, 1].
 */
export function extractTransformedPatch(source, cx, cy, scaleFactor, rotationRad, patchSize, patchCanvas) {
  const ctx = patchCanvas.getContext('2d');
  ctx.clearRect(0, 0, patchSize, patchSize);
  ctx.save();
  ctx.translate(patchSize / 2, patchSize / 2);
  ctx.rotate(rotationRad);
  ctx.scale(1 / scaleFactor, 1 / scaleFactor);
  ctx.translate(-cx, -cy);
  ctx.drawImage(source, 0, 0);
  ctx.restore();

  const px = ctx.getImageData(0, 0, patchSize, patchSize).data;
  const patch = new Float32Array(patchSize * patchSize * 3);
  for (let i = 0; i < patchSize * patchSize; i += 1) {
    patch[i * 3]     = px[i * 4]     / 127.5 - 1;
    patch[i * 3 + 1] = px[i * 4 + 1] / 127.5 - 1;
    patch[i * 3 + 2] = px[i * 4 + 2] / 127.5 - 1;
  }
  return patch;
}

/**
 * Compute the training label for a patch extracted at (cx, cy) with the given scaleFactor.
 * The visible-latitude-extent label encodes geographic scale consistently across maps so
 * the model can compare scale across images with different lat/lon coverage.
 *
 * @param {number} cx - centre pixel x in the source map
 * @param {number} cy - centre pixel y in the source map
 * @param {number} mapWidth - source map width in pixels
 * @param {number} mapHeight - source map height in pixels
 * @param {{ latMin, latMax, lonMin, lonMax }} bounds
 * @param {number} scaleFactor - source pixels per patch pixel
 * @param {number} patchSize - patch side length
 * @returns {Float32Array} [latNorm, lonNorm, scaleNorm] all in [0, 1] relative to map bounds / training range
 */
export function computePatchLabel(cx, cy, mapWidth, mapHeight, bounds, scaleFactor, patchSize) {
  const { latMin, latMax, lonMin, lonMax } = bounds;
  // y=0 is the top of a north-up image (latMax); y=mapHeight is the bottom (latMin).
  const lat = latMax - (cy / mapHeight) * (latMax - latMin);
  const lon = lonMin + (cx / mapWidth)  * (lonMax - lonMin);
  // Normalize to [0, 1] within the map bounds for maximum precision on close-range maps.
  const latNorm = (lat - latMin) / (latMax - latMin);
  const lonNorm = (lon - lonMin) / (lonMax - lonMin);
  const scaleNorm = encodeScaleNorm(scaleFactor);
  return new Float32Array([latNorm, lonNorm, scaleNorm]);
}

/**
 * Decode model output [latNorm, lonNorm, scaleNorm] back to geographic values.
 * All three values are in [0, 1] relative to the map's bounds / training scale range.
 * @param {Float32Array|number[]} output
 * @param {{ latMin, latMax, lonMin, lonMax }} bounds - the map's geographic bounds
 * @returns {{ lat: number, lon: number, scaleFactor: number }}
 */
export function decodePrediction(output, bounds) {
  const [latNorm, lonNorm, scaleNorm] = output;
  return {
    lat:         bounds.latMin + latNorm * (bounds.latMax - bounds.latMin),
    lon:         bounds.lonMin + lonNorm * (bounds.lonMax - bounds.lonMin),
    scaleFactor: decodeScaleNorm(scaleNorm),
  };
}

/**
 * Convert a (lat, lon) coordinate to pixel position on a canvas of the given size,
 * given the map's geographic bounds. Assumes north-up orientation (y=0 is latMax).
 */
export function latLonToPixel(lat, lon, bounds, canvasWidth, canvasHeight) {
  const { latMin, latMax, lonMin, lonMax } = bounds;
  return {
    x: (lon - lonMin) / (lonMax - lonMin) * canvasWidth,
    y: (1 - (lat - latMin) / (latMax - latMin)) * canvasHeight,
  };
}

/**
 * Generate a procedural Voronoi colour map as an ImageBitmap.
 * The map has no geographic significance; lat/lon bounds are supplied separately by the user.
 */
export async function generateProceduralMap(width, height, nCenters = 12) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const centers = Array.from({ length: nCenters }, (_, i) => ({
    x: Math.random() * width,
    y: Math.random() * height,
    hue: (i / nCenters) * 360,
  }));
  const imgData = ctx.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centers.length; c += 1) {
        const dx = x - centers[c].x;
        const dy = y - centers[c].y;
        const d = dx * dx + dy * dy;
        if (d < bestDist) { bestDist = d; best = c; }
      }
      // Convert HSL(hue, 65%, 55%) to RGB
      const h = centers[best].hue / 60;
      const s = 0.65;
      const l = 0.55;
      const c2 = (1 - Math.abs(2 * l - 1)) * s;
      const x2 = c2 * (1 - Math.abs(h % 2 - 1));
      let r = 0; let g = 0; let b = 0;
      if (h < 1)      { r = c2; g = x2; b = 0; }
      else if (h < 2) { r = x2; g = c2; b = 0; }
      else if (h < 3) { r = 0;  g = c2; b = x2; }
      else if (h < 4) { r = 0;  g = x2; b = c2; }
      else if (h < 5) { r = x2; g = 0;  b = c2; }
      else            { r = c2; g = 0;  b = x2; }
      const m = l - c2 / 2;
      const idx = (y * width + x) * 4;
      imgData.data[idx]     = Math.round((r + m) * 255);
      imgData.data[idx + 1] = Math.round((g + m) * 255);
      imgData.data[idx + 2] = Math.round((b + m) * 255);
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const bitmap = await createImageBitmap(canvas);
  return { bitmap, width, height };
}

/**
 * Generate multi-scale training data from an array of map entries.
 * Each entry: { bitmap, width, height, bounds: { latMin, latMax, lonMin, lonMax } }
 *
 * Patches are sampled at random positions and random log-uniform scales within
 * [TRAIN_LOG2_SCALE_MIN, TRAIN_LOG2_SCALE_MAX], with no rotation (north-up).
 * Inference pre-rotates patches by the compass heading before feeding the model,
 * so all patches the model ever sees are north-aligned.
 *
 * @returns {{ inputs: Float32Array[], labels: Float32Array[] }}
 */
export function generateMultiScaleTrainingData(mapEntries, patchSize, samplesPerMap) {
  const patchCanvas = document.createElement('canvas');
  patchCanvas.width = patchSize;
  patchCanvas.height = patchSize;
  const inputs = [];
  const labels = [];

  for (const entry of mapEntries) {
    const { bitmap, width, height, bounds } = entry;
    const half = patchSize / 2;
    for (let s = 0; s < samplesPerMap; s += 1) {
      const cx = half + Math.random() * Math.max(1, width  - patchSize);
      const cy = half + Math.random() * Math.max(1, height - patchSize);
      const log2Scale = TRAIN_LOG2_SCALE_MIN + Math.random() * (TRAIN_LOG2_SCALE_MAX - TRAIN_LOG2_SCALE_MIN);
      const scaleFactor = Math.pow(2, log2Scale);
      const rotationRad = 0;
      const patch = extractTransformedPatch(bitmap, cx, cy, scaleFactor, rotationRad, patchSize, patchCanvas);
      const label = computePatchLabel(cx, cy, width, height, bounds, scaleFactor, patchSize);
      inputs.push(patch);
      labels.push(label);
    }
  }
  return { inputs, labels };
}
