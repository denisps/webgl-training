// Audio feature extraction for EfficientWord-Net-inspired hot-word detection.
// Implements log mel filterbank features matching the original EfficientWord-Net:
//   sampleRate=16000, nfft=512, winlen=0.025s, winstep=0.01s, nfilt=64, preemph=0

export const SAMPLE_RATE = 16000;
export const WINDOW_SAMPLES = 24000;  // 1.5 seconds
export const WIN_SAMPLES = 400;       // 25 ms frame
export const HOP_SAMPLES = 160;       // 10 ms hop
export const N_FFT = 512;
export const N_MEL = 64;
export const N_FRAMES = Math.floor((WINDOW_SAMPLES - WIN_SAMPLES) / HOP_SAMPLES) + 1;  // 148

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildHammingWindow(size) {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (size - 1));
  }
  return win;
}

export function buildMelFilters(sampleRate, nfft, nfilt) {
  const lowMel = hzToMel(0);
  const highMel = hzToMel(sampleRate / 2);
  const halfBins = nfft / 2 + 1;

  // nfilt+2 evenly spaced points in mel scale converted to FFT bin indices
  const bins = new Int32Array(nfilt + 2);
  for (let i = 0; i < nfilt + 2; i++) {
    const hz = melToHz(lowMel + i * (highMel - lowMel) / (nfilt + 1));
    bins[i] = Math.floor((nfft + 1) * hz / sampleRate);
  }

  const filters = [];
  for (let m = 0; m < nfilt; m++) {
    const f = new Float32Array(halfBins);
    for (let k = bins[m]; k < bins[m + 1]; k++) {
      f[k] = (k - bins[m]) / Math.max(bins[m + 1] - bins[m], 1);
    }
    for (let k = bins[m + 1]; k <= bins[m + 2]; k++) {
      f[k] = (bins[m + 2] - k) / Math.max(bins[m + 2] - bins[m + 1], 1);
    }
    filters.push(f);
  }
  return filters;
}

// In-place Cooley-Tukey radix-2 FFT (n must be a power of 2)
function fft(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      let t = real[i]; real[i] = real[j]; real[j] = t;
      t = imag[i]; imag[i] = imag[j]; imag[j] = t;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const angle = -2 * Math.PI / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < n; i += size) {
      let tRe = 1.0;
      let tIm = 0.0;
      for (let k = 0; k < half; k++) {
        const uRe = real[i + k];
        const uIm = imag[i + k];
        const vRe = real[i + k + half] * tRe - imag[i + k + half] * tIm;
        const vIm = real[i + k + half] * tIm + imag[i + k + half] * tRe;
        real[i + k] = uRe + vRe;
        imag[i + k] = uIm + vIm;
        real[i + k + half] = uRe - vRe;
        imag[i + k + half] = uIm - vIm;
        const nextRe = tRe * wRe - tIm * wIm;
        tIm = tRe * wIm + tIm * wRe;
        tRe = nextRe;
      }
    }
  }
}

// Precomputed constants (fixed for the lifetime of the module)
const HAMMING_WINDOW = buildHammingWindow(WIN_SAMPLES);
const MEL_FILTERS = buildMelFilters(SAMPLE_RATE, N_FFT, N_MEL);
const _fftReal = new Float32Array(N_FFT);
const _fftImag = new Float32Array(N_FFT);
const HALF_BINS = N_FFT / 2 + 1;

// Compute time-averaged log mel features for a 1.5 s audio window.
// Returns a Float32Array of length N_MEL (64).
export function extractLogMelFeatures(audioBuffer) {
  let buf = audioBuffer;
  if (buf.length < WINDOW_SAMPLES) {
    const padded = new Float32Array(WINDOW_SAMPLES);
    padded.set(buf);
    buf = padded;
  }

  const result = new Float32Array(N_MEL);

  for (let frame = 0; frame < N_FRAMES; frame++) {
    const offset = frame * HOP_SAMPLES;
    for (let i = 0; i < WIN_SAMPLES; i++) {
      _fftReal[i] = buf[offset + i] * HAMMING_WINDOW[i];
    }
    _fftReal.fill(0, WIN_SAMPLES);
    _fftImag.fill(0);

    fft(_fftReal, _fftImag);

    for (let m = 0; m < N_MEL; m++) {
      const filter = MEL_FILTERS[m];
      let energy = 0;
      for (let k = 0; k < HALF_BINS; k++) {
        energy += filter[k] * (_fftReal[k] * _fftReal[k] + _fftImag[k] * _fftImag[k]) / N_FFT;
      }
      result[m] += Math.log(Math.max(energy, 1e-10));
    }
  }

  for (let m = 0; m < N_MEL; m++) {
    result[m] /= N_FRAMES;
  }
  return result;
}

// RMS energy of an audio buffer (simple voice activity proxy)
export function computeRms(audioBuffer) {
  let sum = 0;
  for (const s of audioBuffer) sum += s * s;
  return Math.sqrt(sum / audioBuffer.length);
}

// Produce an augmented copy of an audio buffer (for training data augmentation)
export function augmentAudio(audioBuffer, timeShift = 0, scale = 1.0, noiseLevel = 0.0) {
  const result = new Float32Array(audioBuffer.length);
  for (let i = 0; i < audioBuffer.length; i++) {
    const src = i - timeShift;
    let sample = (src >= 0 && src < audioBuffer.length) ? audioBuffer[src] : 0;
    sample *= scale;
    if (noiseLevel > 0) sample += (Math.random() * 2 - 1) * noiseLevel;
    result[i] = sample;
  }
  return result;
}

// Generate k mel feature vectors from one audio buffer using random augmentation
export function generateAugmentations(audioBuffer, k = 15) {
  const results = [];
  for (let i = 0; i < k; i++) {
    const shift = Math.round((Math.random() * 2 - 1) * 1600); // ±100 ms
    const scale = 0.7 + Math.random() * 0.6;                  // 0.7 – 1.3×
    const noise = Math.random() * 0.015;
    results.push(extractLogMelFeatures(augmentAudio(audioBuffer, shift, scale, noise)));
  }
  return results;
}

// Create synthetic background mel features (silence, white noise, low hum)
export function generateSyntheticBackground(count = 6) {
  const results = [];
  for (let i = 0; i < count; i++) {
    const audio = new Float32Array(WINDOW_SAMPLES);
    const kind = i % 3;
    if (kind === 1) {
      for (let j = 0; j < audio.length; j++) audio[j] = (Math.random() * 2 - 1) * 0.02;
    } else if (kind === 2) {
      const freq = 50 + Math.random() * 200;
      for (let j = 0; j < audio.length; j++) audio[j] = Math.sin(2 * Math.PI * freq * j / SAMPLE_RATE) * 0.01;
    }
    // kind 0 → silence (all zeros)
    results.push(extractLogMelFeatures(audio));
  }
  return results;
}

// Microphone capture using AudioWorklet at 16 kHz
export class MicrophoneCapture {
  constructor() {
    this._audioContext = null;
    this._stream = null;
    this._workletNode = null;
    this.onChunk = null;  // set to (Float32Array) => void to receive 128-sample chunks
  }

  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    this._audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this._audioContext.audioWorklet.addModule('./worklet.js');
    const source = this._audioContext.createMediaStreamSource(this._stream);
    this._workletNode = new AudioWorkletNode(this._audioContext, 'hotword-processor');
    this._workletNode.port.onmessage = (event) => {
      if (this.onChunk) this.onChunk(event.data);
    };
    const silent = this._audioContext.createGain();
    silent.gain.value = 0;
    source.connect(this._workletNode);
    this._workletNode.connect(silent);
    silent.connect(this._audioContext.destination);
  }

  stop() {
    this._workletNode?.disconnect();
    this._workletNode = null;
    this._stream?.getTracks().forEach((t) => t.stop());
    this._stream = null;
    this._audioContext?.close();
    this._audioContext = null;
  }

  get active() {
    return this._audioContext !== null;
  }
}
