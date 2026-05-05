"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import {
  Play,
  Square,
  Trash2,
  Upload,
  FileText,
  X,
  CircleCheckBig,
  CircleAlert,
  Download,
} from "lucide-react";
import { useAppStore } from "@/lib/app-store";
import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/ui/section-card";
import { pairFiles, type FilePair } from "@/lib/file-pairing";
import { cn } from "@/lib/utils";

type PairState = "pending" | "running" | "done" | "error";

export function BatchQueueSection() {
  const {
    config,
    batchPairs,
    addBatchPairs,
    removeBatchPair,
    clearBatch,
    results,
    addResult,
    pool,
    initPool,
    appendLog,
    setIsRunning,
    isRunning,
    batchProgress,
    setBatchProgress,
    setActiveResultId,
  } = useAppStore();

  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [runtimeStates, setRuntimeStates] = useState<
    Map<string, { state: PairState; error?: string }>
  >(new Map());
  const stopRequestedRef = useRef(false);
  const [unpairedErrors, setUnpairedErrors] = useState<
    Array<{ file: File; reason: string }>
  >([]);

  const intakeFiles = useCallback(
    (files: FileList | File[] | null) => {
      if (!files) return;
      const report = pairFiles(Array.from(files));
      if (report.pairs.length > 0) {
        addBatchPairs(report.pairs);
      }
      setUnpairedErrors(report.unpaired);
      if (report.unpaired.length) {
        for (const u of report.unpaired) {
          appendLog({
            level: "warning",
            message: `Skipped ${u.file.name}: ${u.reason}`,
            timestamp: Date.now(),
          });
        }
      }
    },
    [addBatchPairs, appendLog],
  );

  const totals = useMemo(() => {
    let done = 0;
    let err = 0;
    for (const p of batchPairs) {
      const rt = runtimeStates.get(p.id);
      if (rt?.state === "done") done++;
      else if (rt?.state === "error") err++;
    }
    return { total: batchPairs.length, done, err };
  }, [batchPairs, runtimeStates]);

  const allPending = batchPairs.length > 0 && totals.done + totals.err === 0;

  async function runBatch() {
    if (batchPairs.length === 0) return;
    stopRequestedRef.current = false;
    setIsRunning(true);
    setBatchProgress({
      completed: 0,
      total: batchPairs.length,
      startedAt: Date.now(),
    });
    appendLog({
      level: "info",
      message: `Starting batch of ${batchPairs.length} files.`,
      timestamp: Date.now(),
    });

    try {
      await initPool();
      await pool.prewarm();
    } catch (err) {
      appendLog({
        level: "error",
        message: `Failed to initialize runtime: ${err instanceof Error ? err.message : String(err)}`,
        timestamp: Date.now(),
      });
      setIsRunning(false);
      return;
    }

    const pendingIds = batchPairs
      .filter((p) => runtimeStates.get(p.id)?.state !== "done")
      .map((p) => p.id);
    const pending = batchPairs.filter((p) => pendingIds.includes(p.id));

    let completed = 0;
    const concurrency = Math.max(1, pool.poolSize);
    let cursor = 0;

    async function worker() {
      while (true) {
        if (stopRequestedRef.current) return;
        const i = cursor++;
        if (i >= pending.length) return;
        const pair = pending[i];
        markState(pair.id, { state: "running" });
        setBatchProgress({
          completed,
          total: batchPairs.length,
          currentFile: pair.displayName,
        });
        try {
          const posBytes = await pair.posFile.arrayBuffer();
          const negBytes = await pair.negFile.arrayBuffer();
          const result = await pool.fit({
            jobId: `batch-${Date.now()}`,
            fileIdx: pair.index,
            fileId: `${pair.prefix}-${pair.study}`,
            posBytes,
            negBytes,
            config,
          });
          addResult(pair.id, pair.displayName, result);
          setActiveResultId(pair.id);
          markState(pair.id, { state: "done" });
          appendLog({
            level: "info",
            message: `✓ ${pair.displayName}: f = ${(result.params.f.value / 1e9).toFixed(4)} GHz, α = ${result.params.alpha.value.toExponential(3)} m²/s`,
            timestamp: Date.now(),
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          markState(pair.id, { state: "error", error: msg });
          appendLog({
            level: "error",
            message: `✗ ${pair.displayName}: ${msg}`,
            timestamp: Date.now(),
          });
        } finally {
          completed++;
          setBatchProgress({ completed, total: batchPairs.length });
        }
      }
    }

    await Promise.all(
      Array.from({ length: Math.min(concurrency, pending.length) }, () =>
        worker(),
      ),
    );

    setBatchProgress({
      completed,
      total: batchPairs.length,
      finishedAt: Date.now(),
    });
    setIsRunning(false);
    appendLog({
      level: "info",
      message: `Batch complete: ${completed}/${pending.length} processed.`,
      timestamp: Date.now(),
    });
  }

  function markState(
    id: string,
    value: { state: PairState; error?: string },
  ) {
    setRuntimeStates((prev) => {
      const next = new Map(prev);
      next.set(id, value);
      return next;
    });
  }

  function stop() {
    stopRequestedRef.current = true;
    appendLog({
      level: "warning",
      message: "Stop requested; in-flight fits will finish.",
      timestamp: Date.now(),
    });
  }

  function downloadCsv() {
    if (results.size === 0) return;
    const rows: string[] = [];
    const header = [
      "run_name",
      "start_idx",
      "start_time",
      "grating_spacing_um",
      "A",
      "A_err",
      "B",
      "B_err",
      "C",
      "C_err",
      "alpha",
      "alpha_err",
      "beta",
      "beta_err",
      "theta",
      "theta_err",
      "tau",
      "tau_err",
      "f",
      "f_err",
    ];
    rows.push(header.join(","));
    for (const pair of batchPairs) {
      const entry = results.get(pair.id);
      if (!entry) continue;
      const r = entry.result;
      const line = [
        entry.displayName,
        r.startIdx,
        r.startTime,
        r.gratingSpacing,
        r.params.A.value,
        r.params.A.err,
        r.params.B.value,
        r.params.B.err,
        r.params.C.value,
        r.params.C.err,
        r.params.alpha.value,
        r.params.alpha.err,
        r.params.beta.value,
        r.params.beta.err,
        r.params.theta.value,
        r.params.theta.err,
        r.params.tau.value,
        r.params.tau.err,
        r.params.f.value,
        r.params.f.err,
      ]
        .map((v) => (typeof v === "number" ? v.toString() : `"${v}"`))
        .join(",");
      rows.push(line);
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pytgs-fit-${new Date().toISOString().slice(0, 19)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const hasResults = results.size > 0;

  return (
    <SectionCard
      step={3}
      title="Batch queue"
      subtitle="Drag in POS/NEG pairs. The runtime warms up lazily on the first run."
    >
      <div className="space-y-3">
        <input
          ref={inputRef}
          type="file"
          accept=".txt"
          multiple
          className="hidden"
          onChange={(e) => {
            intakeFiles(e.target.files);
            e.target.value = "";
          }}
        />

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            intakeFiles(e.dataTransfer.files);
          }}
          className={cn(
            "flex cursor-pointer items-center justify-between gap-3 rounded-xl border-2 border-dashed px-4 py-5 transition-colors",
            dragActive
              ? "border-brand bg-brand-soft"
              : "border-rule bg-surface-alt hover:border-brand/60 hover:bg-surface-alt-2",
          )}
          onClick={() => inputRef.current?.click()}
        >
          <div className="flex items-center gap-3">
            <Upload size={20} className="text-psfc-muted" />
            <div>
              <div className="text-sm font-medium text-psfc-ink">
                Drop POS/NEG files here
              </div>
              <div className="text-xs text-psfc-muted">
                Auto-paired by filename:{" "}
                <span className="font-mono">
                  {"{prefix}-{study}-POS-{N}.txt"}
                </span>
              </div>
            </div>
          </div>
          <Button variant="default" size="sm">
            Browse files
          </Button>
        </div>

        {unpairedErrors.length > 0 && (
          <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs">
            <div className="mb-1 font-medium text-amber-900">
              {unpairedErrors.length} file(s) skipped
            </div>
            <ul className="space-y-0.5 text-amber-800">
              {unpairedErrors.slice(0, 5).map((u, i) => (
                <li key={i}>
                  <span className="font-mono">{u.file.name}</span>: {u.reason}
                </li>
              ))}
              {unpairedErrors.length > 5 && (
                <li>… and {unpairedErrors.length - 5} more</li>
              )}
            </ul>
          </div>
        )}

        {batchPairs.length > 0 && (
          <>
            <BatchList
              pairs={batchPairs}
              runtimeStates={runtimeStates}
              onRemove={removeBatchPair}
            />

            <div className="psfc-divider" />

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-psfc-muted">
                <span className="font-medium text-psfc-ink">
                  {totals.done}
                </span>{" "}
                / {totals.total} fitted
                {totals.err > 0 && (
                  <span className="ml-2 text-red-600">
                    · {totals.err} failed
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={downloadCsv}
                  disabled={!hasResults}
                  title="Download fit.csv"
                >
                  <Download size={14} />
                  CSV
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearBatch}
                  disabled={isRunning}
                >
                  <Trash2 size={14} />
                  Clear
                </Button>
                {isRunning ? (
                  <Button variant="danger" onClick={stop}>
                    <Square size={14} />
                    Stop
                  </Button>
                ) : (
                  <Button
                    variant="primary"
                    onClick={runBatch}
                    disabled={!allPending && totals.done === 0 && totals.err === 0 ? false : false}
                  >
                    <Play size={14} />
                    {totals.done + totals.err > 0 ? "Re-run" : "Run batch"}
                  </Button>
                )}
              </div>
            </div>

            {(isRunning || batchProgress.finishedAt) && (
              <ProgressBar
                completed={batchProgress.completed}
                total={batchProgress.total}
                currentFile={batchProgress.currentFile}
                done={!isRunning && !!batchProgress.finishedAt}
              />
            )}
          </>
        )}
      </div>
    </SectionCard>
  );
}

function BatchList({
  pairs,
  runtimeStates,
  onRemove,
}: {
  pairs: FilePair[];
  runtimeStates: Map<string, { state: PairState; error?: string }>;
  onRemove: (id: string) => void;
}) {
  const { activeResultId, setActiveResultId, results, isRunning } =
    useAppStore();

  return (
    <ul className="scrollbar-thin max-h-64 divide-y divide-psfc-rule overflow-auto rounded-md border border-psfc-rule bg-psfc-surface-alt font-mono text-xs">
      {pairs.map((pair) => {
        const rt = runtimeStates.get(pair.id)?.state ?? "pending";
        const hasResult = results.has(pair.id);
        const selected = activeResultId === pair.id;
        return (
          <li
            key={pair.id}
            className={cn(
              "flex cursor-pointer items-center gap-2 px-3 py-1.5 transition-colors",
              selected && "bg-brand-soft",
              !selected && hasResult && "hover:bg-surface-alt",
            )}
            onClick={() => hasResult && setActiveResultId(pair.id)}
          >
            <FileText size={12} className="shrink-0 text-psfc-muted" />
            <span className="min-w-0 flex-1 truncate">
              {pair.displayName}
            </span>
            <StatusDot state={rt} />
            {!isRunning && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(pair.id);
                }}
                className="text-psfc-muted hover:text-psfc-red"
                aria-label="Remove"
              >
                <X size={12} />
              </button>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function StatusDot({ state }: { state: PairState }) {
  if (state === "done")
    return <CircleCheckBig size={12} className="shrink-0 text-green-600" />;
  if (state === "error")
    return <CircleAlert size={12} className="shrink-0 text-red-600" />;
  if (state === "running")
    return (
      <span className="inline-block h-2 w-2 shrink-0 animate-pulse rounded-full bg-brand" />
    );
  return (
    <span className="inline-block h-2 w-2 shrink-0 rounded-full bg-rule" />
  );
}

function ProgressBar({
  completed,
  total,
  currentFile,
  done,
}: {
  completed: number;
  total: number;
  currentFile?: string;
  done: boolean;
}) {
  const pct = total > 0 ? (completed / total) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="h-1.5 overflow-hidden rounded-full bg-rule">
        <div
          className={cn(
            "h-full transition-all",
            done ? "bg-emerald-500" : "bg-brand",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-psfc-muted">
        <span>
          {completed}/{total} complete
          {currentFile && !done && (
            <span className="ml-2 font-mono text-psfc-body">{currentFile}</span>
          )}
        </span>
        <span>{pct.toFixed(0)}%</span>
      </div>
    </div>
  );
}
