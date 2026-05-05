"use client";

import type {
  CalibrationResult,
  FitResult,
  LogLevel,
  PyTGSConfig,
  WorkerLog,
} from "./config-types";

type WorkerIncoming =
  | { id: string; type: "ready" }
  | { id: string; type: "progress"; stage: string }
  | { id: string; type: "result"; result: unknown }
  | { id: string; type: "error"; message: string }
  | { type: "log"; level: LogLevel; message: string };

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  onProgress?: (stage: string) => void;
}

export interface FitRequest {
  jobId: string;
  fileIdx: number;
  fileId: string;
  posBytes: ArrayBuffer;
  negBytes: ArrayBuffer;
  config: PyTGSConfig;
  onProgress?: (stage: string) => void;
}

export interface CalibrateRequest {
  posBytes: ArrayBuffer;
  negBytes: ArrayBuffer;
  config: PyTGSConfig;
  soundSpeed: number;
  onProgress?: (stage: string) => void;
}

let _nextId = 0;
const nextId = () => `r${++_nextId}`;

function toErrorString(v: unknown): string {
  if (v == null) return "Unknown error";
  if (typeof v === "string") return v;
  if (v instanceof Error) return v.message || v.toString();
  if (typeof v === "object") {
    const rec = v as Record<string, unknown>;
    if (typeof rec.message === "string") return rec.message;
    try {
      return JSON.stringify(rec);
    } catch {
      /* fall through */
    }
  }
  try {
    return String(v);
  } catch {
    return "Unserializable error";
  }
}

class PyodideWorker {
  readonly worker: Worker;
  readonly pending = new Map<string, PendingRequest>();
  readonly logHandlers = new Set<(log: WorkerLog) => void>();
  ready = false;
  private readyPromise: Promise<void> | null = null;

  constructor() {
    this.worker = new Worker("/workers/pyodide.worker.js");
    this.worker.onmessage = (e: MessageEvent<WorkerIncoming>) => {
      const msg = e.data;
      if (msg.type === "log") {
        const log: WorkerLog = {
          level: msg.level,
          message: msg.message,
          timestamp: Date.now(),
        };
        for (const h of this.logHandlers) h(log);
        return;
      }
      const req = this.pending.get(msg.id);
      if (!req) return;
      if (msg.type === "ready") {
        req.resolve(undefined);
        this.pending.delete(msg.id);
      } else if (msg.type === "progress") {
        req.onProgress?.(msg.stage);
      } else if (msg.type === "result") {
        req.resolve(msg.result);
        this.pending.delete(msg.id);
      } else if (msg.type === "error") {
        req.reject(new Error(toErrorString(msg.message)));
        this.pending.delete(msg.id);
      }
    };
    this.worker.onerror = (ev) => {
      const message = ev?.message || toErrorString(ev);
      for (const h of this.logHandlers) {
        h({
          level: "error",
          message: `Worker runtime error: ${message}`,
          timestamp: Date.now(),
        });
      }
      // Reject anything still in flight so callers don't hang.
      for (const [, req] of this.pending) {
        req.reject(new Error(`Worker crashed: ${message}`));
      }
      this.pending.clear();
    };
    this.worker.onmessageerror = (ev) => {
      const message = toErrorString(ev);
      for (const h of this.logHandlers) {
        h({
          level: "error",
          message: `Worker message serialization error: ${message}`,
          timestamp: Date.now(),
        });
      }
    };
  }

  init(): Promise<void> {
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = new Promise<void>((resolve, reject) => {
      const id = nextId();
      this.pending.set(id, {
        resolve: () => {
          this.ready = true;
          resolve();
        },
        reject,
      });
      this.worker.postMessage({ id, type: "init" });
    });
    return this.readyPromise;
  }

  fit(req: FitRequest): Promise<FitResult> {
    const id = nextId();
    return new Promise<FitResult>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (v) => resolve(v as FitResult),
        reject,
        onProgress: req.onProgress,
      });
      this.worker.postMessage({
        id,
        type: "fit",
        jobId: req.jobId,
        fileIdx: req.fileIdx,
        fileId: req.fileId,
        posBytes: req.posBytes,
        negBytes: req.negBytes,
        config: req.config,
      });
    });
  }

  calibrate(req: CalibrateRequest): Promise<CalibrationResult> {
    const id = nextId();
    return new Promise<CalibrationResult>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (v) => resolve(v as CalibrationResult),
        reject,
        onProgress: req.onProgress,
      });
      this.worker.postMessage({
        id,
        type: "calibrate",
        posBytes: req.posBytes,
        negBytes: req.negBytes,
        config: req.config,
        soundSpeed: req.soundSpeed,
      });
    });
  }

  terminate() {
    this.worker.terminate();
    this.pending.clear();
  }
}

export class PyodideWorkerPool {
  private workers: PyodideWorker[] = [];
  private roundRobin = 0;
  private initialized = false;
  private logHandlers = new Set<(log: WorkerLog) => void>();

  constructor(private size: number = defaultPoolSize()) {}

  onLog(handler: (log: WorkerLog) => void): () => void {
    this.logHandlers.add(handler);
    for (const w of this.workers) w.logHandlers.add(handler);
    return () => {
      this.logHandlers.delete(handler);
      for (const w of this.workers) w.logHandlers.delete(handler);
    };
  }

  // Initializes the pool. By default we eagerly initialize just one worker,
  // so the first user action is responsive. Additional workers are spun up
  // lazily on demand by `prewarm(n)` or on first concurrent use.
  async init(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;
    const first = this.spawn();
    await first.init();
  }

  async prewarm(count: number = this.size): Promise<void> {
    await this.init();
    const target = Math.min(count, this.size);
    while (this.workers.length < target) {
      this.spawn();
    }
    await Promise.all(this.workers.map((w) => w.init()));
  }

  private spawn(): PyodideWorker {
    const w = new PyodideWorker();
    for (const h of this.logHandlers) w.logHandlers.add(h);
    this.workers.push(w);
    return w;
  }

  private pickWorker(): PyodideWorker {
    // Ensure at least one worker exists.
    if (this.workers.length === 0) this.spawn();
    // If we haven't filled the pool yet and the next slot would block, grow it.
    if (this.workers.length < this.size) {
      const candidate = this.workers[this.roundRobin % this.workers.length];
      if (candidate.pending.size > 0) {
        this.spawn();
      }
    }
    const idx = this.roundRobin % this.workers.length;
    this.roundRobin = (this.roundRobin + 1) % Math.max(this.workers.length, 1);
    return this.workers[idx];
  }

  async fit(req: FitRequest): Promise<FitResult> {
    await this.init();
    const w = this.pickWorker();
    if (!w.ready) await w.init();
    return w.fit(req);
  }

  async calibrate(req: CalibrateRequest): Promise<CalibrationResult> {
    await this.init();
    const w = this.workers[0] ?? this.spawn();
    if (!w.ready) await w.init();
    return w.calibrate(req);
  }

  get workerCount(): number {
    return this.workers.length;
  }

  get poolSize(): number {
    return this.size;
  }

  terminate() {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.initialized = false;
  }
}

function defaultPoolSize(): number {
  if (typeof navigator === "undefined") return 4;
  const hc = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(hc - 1, 8));
}

// Module-level singleton so every component shares the same pool.
let _poolSingleton: PyodideWorkerPool | null = null;

export function getWorkerPool(): PyodideWorkerPool {
  if (!_poolSingleton) _poolSingleton = new PyodideWorkerPool();
  return _poolSingleton;
}
