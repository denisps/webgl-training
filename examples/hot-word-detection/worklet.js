// AudioWorklet processor: forwards audio chunks from the audio pipeline to the main thread.
class HotwordProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel?.length) {
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}

registerProcessor('hotword-processor', HotwordProcessor);
