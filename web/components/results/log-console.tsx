"use client";

import { useEffect, useRef } from "react";
import { Trash2 } from "lucide-react";
import { useAppStore } from "@/lib/app-store";
import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/ui/section-card";
import { cn } from "@/lib/utils";

export function LogConsole() {
  const { logs, clearLogs } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [logs.length]);

  return (
    <SectionCard
      title="Log"
      subtitle={`${logs.length} ${logs.length === 1 ? "entry" : "entries"}`}
      headerRight={
        logs.length > 0 ? (
          <Button variant="ghost" size="sm" onClick={clearLogs}>
            <Trash2 size={14} />
            Clear
          </Button>
        ) : null
      }
      bodyClassName="p-0"
    >
      <div
        ref={scrollRef}
        className="scrollbar-thin max-h-48 overflow-auto bg-[#0f1117] font-mono text-xs text-gray-200"
      >
        {logs.length === 0 ? (
          <div className="p-4 text-gray-500">
            No log output yet. Calibration and batch runs appear here.
          </div>
        ) : (
          <ul className="divide-y divide-white/5">
            {logs.map((l, i) => (
              <li
                key={`${l.timestamp}-${i}`}
                className={cn(
                  "flex gap-2 px-3 py-1",
                  l.level === "error" && "bg-red-950/30 text-red-300",
                  l.level === "warning" && "text-amber-300",
                )}
              >
                <span className="shrink-0 text-gray-500">
                  {new Date(l.timestamp).toLocaleTimeString([], {
                    hour12: false,
                  })}
                </span>
                <span className="flex-1 whitespace-pre-wrap break-words">
                  {l.message}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </SectionCard>
  );
}
