// Real-time ONNX inference engine for Benoit in Node.js / TypeScript.
// Ingests public match events into a ring-buffer and executes benoit.onnx.

import * as path from "node:path";
import * as fs from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

export const PRIVATE_DIM = 57;
export const EVENT_DIM = 38;
export const EVENT_HISTORY_LEN = 32;

// Event ring buffer matching Python/Rust implementations
export class EventBuffer {
  private buffer: Float32Array[] = [];
  private capacity: number;

  constructor(capacity = EVENT_HISTORY_LEN) {
    this.capacity = capacity;
  }

  clear(): void {
    this.buffer = [];
  }

  recordAuctionResult(winnerIdx: number | null, letter: string, paid: number): void {
    const ev = new Float32Array(EVENT_DIM);
    ev[1] = 1.0; // AUCTION_RESULT type

    if (winnerIdx !== null && winnerIdx >= 0 && winnerIdx < 6) {
      ev[3 + winnerIdx] = 1.0;
    } else {
      ev[3 + 6] = 1.0; // UNSOLD
    }

    const upper = letter.toUpperCase();
    const ci = upper.charCodeAt(0) - 65;
    if (ci >= 0 && ci < 26) {
      ev[10 + ci] = 1.0;
    }

    ev[36] = paid / 50.0;
    ev[37] = 0.0;

    this.push(ev);
  }

  recordWordPlay(playerIdx: number, word: string, reward: number): void {
    const ev = new Float32Array(EVENT_DIM);
    ev[2] = 1.0; // WORD_PLAY type

    if (playerIdx >= 0 && playerIdx < 6) {
      ev[3 + playerIdx] = 1.0;
    }

    for (let i = 0; i < word.length; i++) {
      const ci = word.toUpperCase().charCodeAt(i) - 65;
      if (ci >= 0 && ci < 26) {
        ev[10 + ci] += 1.0 / 5.0;
      }
    }

    ev[36] = reward / 50.0;
    ev[37] = word.length / 10.0;

    this.push(ev);
  }

  private push(event: Float32Array): void {
    if (this.buffer.length >= this.capacity) {
      this.buffer.shift();
    }
    this.buffer.push(event);
  }

  toFlatArray(): Float32Array {
    const out = new Float32Array(this.capacity * EVENT_DIM);
    const offset = Math.max(0, this.capacity - this.buffer.length);
    for (let i = 0; i < this.buffer.length; i++) {
      out.set(this.buffer[i]!, (offset + i) * EVENT_DIM);
    }
    return out;
  }
}

export interface ModelPrediction {
  bidFraction: number;
  playThreshold: number;
  valueEstimate: number;
}

export class BenoitModelSession {
  private session: ort.InferenceSession | null = null;
  public eventBuffer: EventBuffer;
  public isReady = false;
  private lastPrediction: ModelPrediction | null = null;

  constructor() {
    this.eventBuffer = new EventBuffer(EVENT_HISTORY_LEN);
    this.initSession();
  }

  private async initSession(): Promise<void> {
    try {
      const currentDir = path.dirname(fileURLToPath(import.meta.url));
      const candidates = [
        path.join(process.cwd(), "roger.onnx"),
        path.join(process.cwd(), "benoit.onnx"),
        path.join(currentDir, "..", "roger.onnx"),
        path.join(currentDir, "..", "benoit.onnx"),
        path.join(currentDir, "..", "..", "rust-client", "roger.onnx"),
        path.join(currentDir, "..", "..", "rust-client", "benoit.onnx"),
      ];

      let modelPath = candidates.find((p) => fs.existsSync(p));
      if (!modelPath) {
        console.warn("[benoit-model] No benoit.onnx file found. Falling back to calibrated heuristics.");
        return;
      }

      this.session = await ort.InferenceSession.create(modelPath);
      this.isReady = true;
      console.log(`[benoit-model] ONNX model successfully loaded from ${modelPath}`);
    } catch (err) {
      console.error("[benoit-model] Failed to initialize ONNX runtime session:", err);
    }
  }

  getLastPrediction(): ModelPrediction | null {
    return this.lastPrediction;
  }

  buildPrivateState(params: {
    letter: string;
    myBalance: number;
    myRack: Map<string, number>;
    roundNum?: number;
    bestWordScore?: number;
    bestWordLen?: number;
    bestWordBonus?: number;
  }): Float32Array {
    const priv = new Float32Array(PRIVATE_DIM);
    priv[0] = params.myBalance / 100.0;

    for (const [char, count] of params.myRack.entries()) {
      const ci = char.toUpperCase().charCodeAt(0) - 65;
      if (ci >= 0 && ci < 26) {
        priv[1 + ci] = count / 5.0;
      }
    }

    const cli = params.letter.toUpperCase().charCodeAt(0) - 65;
    if (cli >= 0 && cli < 26) {
      priv[27 + cli] = 1.0;
    }

    if (params.bestWordScore !== undefined) {
      priv[53] = params.bestWordScore / 50.0;
    }
    if (params.bestWordLen !== undefined) {
      priv[54] = params.bestWordLen / 10.0;
    }
    if (params.bestWordBonus !== undefined) {
      priv[55] = params.bestWordBonus / 30.0;
    }

    priv[56] = Math.min((params.roundNum ?? 1) / 70.0, 1.0);
    return priv;
  }

  async predict(params: {
    letter: string;
    myBalance: number;
    myRack: Map<string, number>;
    roundNum?: number;
    bestWordScore?: number;
    bestWordLen?: number;
    bestWordBonus?: number;
  }): Promise<ModelPrediction | null> {
    if (!this.session) {
      return null;
    }

    try {
      const privateData = this.buildPrivateState(params);
      const eventsData = this.eventBuffer.toFlatArray();

      const privTensor = new ort.Tensor("float32", privateData, [1, PRIVATE_DIM]);
      const eventsTensor = new ort.Tensor("float32", eventsData, [1, EVENT_HISTORY_LEN, EVENT_DIM]);

      const feeds = { private: privTensor, events: eventsTensor };
      const results = await this.session.run(feeds);
      const outTensor = results.output || Object.values(results)[0];
      const data = outTensor?.data as Float32Array | number[];

      if (!data || data.length < 2) {
        return null;
      }

      const bidFraction = Math.max(0.0, Math.min(1.0, data[0]!));
      const playThreshold = Math.max(0.0, Math.min(1.0, data[1]!));
      const valueEstimate = data[2] ?? 0.0;

      this.lastPrediction = { bidFraction, playThreshold, valueEstimate };
      return this.lastPrediction;
    } catch (err) {
      console.warn("[benoit-model] Inference error:", err);
      return null;
    }
  }
}

// Global singleton instance
export const benoitModel = new BenoitModelSession();
