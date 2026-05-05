"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  CalibrationResult,
  FitResult,
  PyTGSConfig,
  WorkerLog,
} from "./config-types";
import type { FilePair } from "./file-pairing";
import { DEFAULT_CONFIG } from "./default-config";
import { getWorkerPool, type PyodideWorkerPool } from "./pyodide-pool";

type PoolStatus = "idle" | "loading" | "ready" | "error";

interface BatchProgress {
  completed: number;
  total: number;
  currentFile?: string;
  startedAt?: number;
  finishedAt?: number;
}

interface FilePairRef {
  pos: File | null;
  neg: File | null;
}

interface ResultEntry {
  pairId: string;
  displayName: string;
  result: FitResult;
  fittedAt: number;
}

interface AppStoreValue {
  config: PyTGSConfig;
  setConfig: (c: PyTGSConfig) => void;
  updateConfig: (fn: (c: PyTGSConfig) => PyTGSConfig) => void;

  calibrationFiles: FilePairRef;
  setCalibrationFiles: (pair: FilePairRef) => void;
  calibrationResult: CalibrationResult | null;
  setCalibrationResult: (r: CalibrationResult | null) => void;

  baselineFiles: FilePairRef;
  setBaselineFiles: (pair: FilePairRef) => void;

  batchPairs: FilePair[];
  setBatchPairs: (p: FilePair[]) => void;
  addBatchPairs: (p: FilePair[]) => void;
  removeBatchPair: (id: string) => void;
  clearBatch: () => void;

  results: Map<string, ResultEntry>;
  addResult: (pairId: string, displayName: string, result: FitResult) => void;
  clearResults: () => void;
  activeResultId: string | null;
  setActiveResultId: (id: string | null) => void;

  logs: WorkerLog[];
  appendLog: (log: WorkerLog) => void;
  clearLogs: () => void;

  poolStatus: PoolStatus;
  poolError: string | null;
  workerCount: number;
  poolSize: number;
  initPool: () => Promise<void>;

  isRunning: boolean;
  setIsRunning: (v: boolean) => void;
  batchProgress: BatchProgress;
  setBatchProgress: (p: BatchProgress) => void;

  pool: PyodideWorkerPool;
}

const AppStoreContext = createContext<AppStoreValue | null>(null);

const MAX_LOGS = 500;

export function AppStoreProvider({ children }: { children: React.ReactNode }) {
  const poolRef = useRef<PyodideWorkerPool | null>(null);
  if (poolRef.current === null) {
    poolRef.current = getWorkerPool();
  }
  const pool = poolRef.current;

  const [config, setConfigState] = useState<PyTGSConfig>(() =>
    structuredClone(DEFAULT_CONFIG),
  );
  const [calibrationFiles, setCalibrationFiles] = useState<FilePairRef>({
    pos: null,
    neg: null,
  });
  const [calibrationResult, setCalibrationResult] =
    useState<CalibrationResult | null>(null);
  const [baselineFiles, setBaselineFiles] = useState<FilePairRef>({
    pos: null,
    neg: null,
  });
  const [batchPairs, setBatchPairsState] = useState<FilePair[]>([]);
  const [results, setResults] = useState<Map<string, ResultEntry>>(new Map());
  const [activeResultId, setActiveResultId] = useState<string | null>(null);
  const [logs, setLogs] = useState<WorkerLog[]>([]);
  const [poolStatus, setPoolStatus] = useState<PoolStatus>("idle");
  const [poolError, setPoolError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<BatchProgress>({
    completed: 0,
    total: 0,
  });
  const [workerCount, setWorkerCount] = useState(0);

  useEffect(() => {
    const unsub = pool.onLog((log) => {
      setLogs((prev) => {
        const next = [...prev, log];
        if (next.length > MAX_LOGS) next.splice(0, next.length - MAX_LOGS);
        return next;
      });
    });
    return () => {
      unsub();
    };
  }, [pool]);

  const updateConfig = useCallback(
    (fn: (c: PyTGSConfig) => PyTGSConfig) => {
      setConfigState((prev) => fn(structuredClone(prev)));
    },
    [],
  );

  const setConfig = useCallback((c: PyTGSConfig) => {
    setConfigState(structuredClone(c));
  }, []);

  const setBatchPairs = useCallback((p: FilePair[]) => {
    setBatchPairsState(p);
  }, []);

  const addBatchPairs = useCallback((p: FilePair[]) => {
    setBatchPairsState((prev) => {
      const existingIds = new Set(prev.map((x) => x.id));
      const additions = p.filter((x) => !existingIds.has(x.id));
      return [...prev, ...additions];
    });
  }, []);

  const removeBatchPair = useCallback((id: string) => {
    setBatchPairsState((prev) => prev.filter((x) => x.id !== id));
  }, []);

  const clearBatch = useCallback(() => {
    setBatchPairsState([]);
  }, []);

  const addResult = useCallback(
    (pairId: string, displayName: string, result: FitResult) => {
      setResults((prev) => {
        const next = new Map(prev);
        next.set(pairId, {
          pairId,
          displayName,
          result,
          fittedAt: Date.now(),
        });
        return next;
      });
      setActiveResultId((prev) => prev ?? pairId);
    },
    [],
  );

  const clearResults = useCallback(() => {
    setResults(new Map());
    setActiveResultId(null);
  }, []);

  const appendLog = useCallback((log: WorkerLog) => {
    setLogs((prev) => {
      const next = [...prev, log];
      if (next.length > MAX_LOGS) next.splice(0, next.length - MAX_LOGS);
      return next;
    });
  }, []);

  const clearLogs = useCallback(() => setLogs([]), []);

  const initPool = useCallback(async () => {
    if (poolStatus === "ready" || poolStatus === "loading") return;
    setPoolStatus("loading");
    setPoolError(null);
    try {
      await pool.init();
      setPoolStatus("ready");
      setWorkerCount(pool.workerCount);
      // Prewarm the rest of the pool in the background; swallow errors so
      // one flaky worker doesn't reject globally.
      pool
        .prewarm()
        .then(() => setWorkerCount(pool.workerCount))
        .catch((err) => {
          appendLog({
            level: "warning",
            message: `Prewarm failed for a worker: ${err instanceof Error ? err.message : String(err)}`,
            timestamp: Date.now(),
          });
        });
    } catch (err) {
      setPoolStatus("error");
      setPoolError(err instanceof Error ? err.message : String(err));
    }
  }, [pool, poolStatus, appendLog]);

  const value = useMemo<AppStoreValue>(
    () => ({
      config,
      setConfig,
      updateConfig,
      calibrationFiles,
      setCalibrationFiles,
      calibrationResult,
      setCalibrationResult,
      baselineFiles,
      setBaselineFiles,
      batchPairs,
      setBatchPairs,
      addBatchPairs,
      removeBatchPair,
      clearBatch,
      results,
      addResult,
      clearResults,
      activeResultId,
      setActiveResultId,
      logs,
      appendLog,
      clearLogs,
      poolStatus,
      poolError,
      workerCount,
      poolSize: pool.poolSize,
      initPool,
      isRunning,
      setIsRunning,
      batchProgress,
      setBatchProgress,
      pool,
    }),
    [
      config,
      setConfig,
      updateConfig,
      calibrationFiles,
      calibrationResult,
      baselineFiles,
      batchPairs,
      setBatchPairs,
      addBatchPairs,
      removeBatchPair,
      clearBatch,
      results,
      addResult,
      clearResults,
      activeResultId,
      logs,
      appendLog,
      clearLogs,
      poolStatus,
      poolError,
      workerCount,
      pool,
      initPool,
      isRunning,
      batchProgress,
    ],
  );

  return (
    <AppStoreContext.Provider value={value}>
      {children}
    </AppStoreContext.Provider>
  );
}

export function useAppStore(): AppStoreValue {
  const ctx = useContext(AppStoreContext);
  if (!ctx) {
    throw new Error("useAppStore must be used within AppStoreProvider");
  }
  return ctx;
}
