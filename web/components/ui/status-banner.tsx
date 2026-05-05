"use client";

import { useAppStore } from "@/lib/app-store";
import { Loader2, CheckCircle2, CircleAlert, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

export function StatusBanner() {
  const { poolStatus, poolError, workerCount, poolSize } = useAppStore();

  if (poolStatus === "idle") {
    return null;
  }

  return (
    <div
      className={cn(
        "psfc-container flex items-center gap-3 py-2 text-sm",
        poolStatus === "loading" && "text-psfc-body",
        poolStatus === "ready" && "text-green-700",
        poolStatus === "error" && "text-red-700",
      )}
    >
      {poolStatus === "loading" && (
        <>
          <Loader2 size={16} className="animate-spin" />
          <span>
            Loading scientific Python runtime… the first load is ~30 MB and
            caches for instant subsequent visits.
          </span>
        </>
      )}
      {poolStatus === "ready" && (
        <>
          <CheckCircle2 size={16} className="text-green-600" />
          <span>Runtime ready.</span>
          <span className="inline-flex items-center gap-1 text-psfc-muted">
            <Cpu size={14} />
            {workerCount}/{poolSize} workers warm
          </span>
        </>
      )}
      {poolStatus === "error" && (
        <>
          <CircleAlert size={16} />
          <span>Failed to load runtime: {poolError}</span>
        </>
      )}
    </div>
  );
}
