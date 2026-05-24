function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const options = [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q],
  ][i % 6];
  return options.map((value) => Math.round(value * 255));
}

export function generateMap(width, height, nRegions) {
  const centers = Array.from({ length: nRegions }, (_, index) => ({
    x: Math.random() * width,
    y: Math.random() * height,
    color: hsvToRgb(index / nRegions, 0.65, 0.92),
  }));
  const imageData = new Uint8ClampedArray(width * height * 4);
  const regionMask = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let bestRegion = 0;
      let bestDistance = Infinity;
      for (let region = 0; region < centers.length; region += 1) {
        const dx = x - centers[region].x;
        const dy = y - centers[region].y;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          bestDistance = distance;
          bestRegion = region;
        }
      }
      const [r, g, b] = centers[bestRegion].color;
      const index = y * width + x;
      imageData[index * 4 + 0] = r;
      imageData[index * 4 + 1] = g;
      imageData[index * 4 + 2] = b;
      imageData[index * 4 + 3] = 255;
      regionMask[index] = bestRegion;
    }
  }

  return { imageData, regionMask };
}

export function extractPatch(imageData, mapWidth, x, y, patchSize) {
  const half = Math.floor(patchSize / 2);
  const patch = new Float32Array(patchSize * patchSize * 3);
  const mapHeight = imageData.length / (mapWidth * 4);
  const rgbMidpoint = 127.5;
  for (let py = 0; py < patchSize; py += 1) {
    for (let px = 0; px < patchSize; px += 1) {
      const srcX = Math.max(0, Math.min(mapWidth - 1, x + px - half));
      const srcY = Math.max(0, Math.min(mapHeight - 1, y + py - half));
      const srcIndex = (srcY * mapWidth + srcX) * 4;
      const dstIndex = (py * patchSize + px) * 3;
      patch[dstIndex + 0] = imageData[srcIndex + 0] / rgbMidpoint - 1;
      patch[dstIndex + 1] = imageData[srcIndex + 1] / rgbMidpoint - 1;
      patch[dstIndex + 2] = imageData[srcIndex + 2] / rgbMidpoint - 1;
    }
  }
  return patch;
}

/**
 * Divide an image into a nRegions-cell grid, choosing the column/row layout
 * that best matches the image aspect ratio. Returns the region mask and the
 * grid dimensions so callers can sample patches without a full-image scan.
 */
export function assignRegionsByGrid(width, height, nRegions) {
  let bestCols = 1;
  let bestScore = Infinity;
  for (let cols = 1; cols <= nRegions; cols += 1) {
    if (nRegions % cols !== 0) continue;
    const rows = nRegions / cols;
    const score = Math.abs(Math.log((cols / rows) / (width / height)));
    if (score < bestScore) {
      bestScore = score;
      bestCols = cols;
    }
  }
  const gridCols = bestCols;
  const gridRows = nRegions / gridCols;
  const regionMask = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const row = Math.min(gridRows - 1, Math.floor((y * gridRows) / height));
    for (let x = 0; x < width; x += 1) {
      const col = Math.min(gridCols - 1, Math.floor((x * gridCols) / width));
      regionMask[y * width + x] = row * gridCols + col;
    }
  }
  return { regionMask, gridCols, gridRows };
}

/**
 * Fast patch sampling that avoids iterating all pixels: samples directly
 * from each grid cell's bounding box. Use this when the image is large.
 */
export function generateGridTrainingPatches(imageData, width, height, gridCols, gridRows, patchSize, patchesPerRegion) {
  const inputs = [];
  const labels = [];
  const cellW = width / gridCols;
  const cellH = height / gridRows;
  const half = Math.floor(patchSize / 2);
  for (let row = 0; row < gridRows; row += 1) {
    for (let col = 0; col < gridCols; col += 1) {
      const region = row * gridCols + col;
      const xMin = Math.ceil(col * cellW) + half;
      const xMax = Math.floor((col + 1) * cellW) - half;
      const yMin = Math.ceil(row * cellH) + half;
      const yMax = Math.floor((row + 1) * cellH) - half;
      if (xMax <= xMin || yMax <= yMin) continue;
      for (let sample = 0; sample < patchesPerRegion; sample += 1) {
        const x = xMin + Math.floor(Math.random() * (xMax - xMin));
        const y = yMin + Math.floor(Math.random() * (yMax - yMin));
        const patch = extractPatch(imageData, width, x, y, patchSize);
        const jitter = 1 + (Math.random() * 0.16 - 0.08);
        for (let index = 0; index < patch.length; index += 1) {
          patch[index] = Math.max(-1, Math.min(1, patch[index] * jitter + (Math.random() * 0.04 - 0.02)));
        }
        inputs.push(patch);
        labels.push(region);
      }
    }
  }
  return { inputs, labels };
}

export function generateTrainingPatches(imageData, regionMask, mapWidth, mapHeight, patchSize, patchesPerRegion) {
  const coordsByRegion = new Map();
  for (let y = 0; y < mapHeight; y += 1) {
    for (let x = 0; x < mapWidth; x += 1) {
      const region = regionMask[y * mapWidth + x];
      if (!coordsByRegion.has(region)) {
        coordsByRegion.set(region, []);
      }
      coordsByRegion.get(region).push({ x, y });
    }
  }

  const inputs = [];
  const labels = [];
  for (const [region, coords] of coordsByRegion.entries()) {
    for (let sample = 0; sample < patchesPerRegion; sample += 1) {
      const point = coords[Math.floor(Math.random() * coords.length)];
      const patch = extractPatch(imageData, mapWidth, point.x, point.y, patchSize);
      const jitter = 1 + (Math.random() * 0.16 - 0.08);
      for (let index = 0; index < patch.length; index += 1) {
        patch[index] = Math.max(-1, Math.min(1, patch[index] * jitter + (Math.random() * 0.04 - 0.02)));
      }
      inputs.push(patch);
      labels.push(region);
    }
  }
  return { inputs, labels };
}
