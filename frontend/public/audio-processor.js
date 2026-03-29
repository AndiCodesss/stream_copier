class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const requestedFrameCount = Number(options?.processorOptions?.targetFrameCount ?? 2048);
    this.targetFrameCount = Number.isFinite(requestedFrameCount) ? Math.max(256, Math.floor(requestedFrameCount)) : 2048;
    this.pending = new Int16Array(this.targetFrameCount);
    this.pendingLength = 0;
  }

  ensureCapacity(nextLength) {
    if (nextLength <= this.pending.length) {
      return;
    }

    let capacity = this.pending.length;
    while (capacity < nextLength) {
      capacity *= 2;
    }

    const expanded = new Int16Array(capacity);
    expanded.set(this.pending.subarray(0, this.pendingLength));
    this.pending = expanded;
  }

  emitFrames() {
    while (this.pendingLength >= this.targetFrameCount) {
      const samples = new Int16Array(this.targetFrameCount);
      samples.set(this.pending.subarray(0, this.targetFrameCount));
      this.port.postMessage({ samples: samples.buffer }, [samples.buffer]);
      this.pending.copyWithin(0, this.targetFrameCount, this.pendingLength);
      this.pendingLength -= this.targetFrameCount;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }

    const channel = input[0];
    this.ensureCapacity(this.pendingLength + channel.length);
    const startOffset = this.pendingLength;

    for (let index = 0; index < channel.length; index += 1) {
      const value = Math.max(-1, Math.min(1, channel[index]));
      this.pending[startOffset + index] = value < 0 ? value * 32768 : value * 32767;
    }

    this.pendingLength += channel.length;
    this.emitFrames();
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
