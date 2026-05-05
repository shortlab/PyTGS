"use client";

import { useAppStore } from "@/lib/app-store";
import type { FitParamName } from "@/lib/config-types";
import { SectionCard } from "@/components/ui/section-card";
import { cn } from "@/lib/utils";

const PARAM_ORDER: FitParamName[] = [
  "A",
  "B",
  "C",
  "alpha",
  "beta",
  "theta",
  "tau",
  "f",
];

const LABELS: Record<FitParamName, string> = {
  A: "A — thermal amplitude",
  B: "B — acoustic amplitude",
  C: "C — signal offset",
  alpha: "α — thermal diffusivity",
  beta: "β — disp./reflectance ratio",
  theta: "θ — acoustic phase",
  tau: "τ — acoustic decay time",
  f: "f — SAW frequency",
};

export function ResultsTable() {
  const { results, activeResultId } = useAppStore();
  const active = activeResultId ? results.get(activeResultId) : null;

  return (
    <SectionCard
      title="Fit parameters"
      subtitle={
        active
          ? active.displayName
          : "Select a result from the batch queue to inspect."
      }
    >
      {!active ? (
        <div className="flex h-48 items-center justify-center text-sm text-psfc-muted">
          No result selected.
        </div>
      ) : (
        <div className="overflow-hidden rounded-md border border-psfc-rule">
          <table className="w-full text-sm">
            <thead className="bg-psfc-surface-alt">
              <tr className="text-left">
                <th className="px-3 py-2 font-medium text-psfc-muted">
                  Parameter
                </th>
                <th className="px-3 py-2 text-right font-medium text-psfc-muted">
                  Value
                </th>
                <th className="px-3 py-2 text-right font-medium text-psfc-muted">
                  1σ error
                </th>
                <th className="px-3 py-2 text-right font-medium text-psfc-muted">
                  Unit
                </th>
              </tr>
            </thead>
            <tbody className="font-mono text-xs">
              <tr className="border-t border-psfc-rule">
                <td className="px-3 py-1.5 text-sm text-psfc-ink">
                  <span className="font-sans">Start time</span>
                </td>
                <td className="px-3 py-1.5 text-right">
                  {active.result.startTime.toExponential(4)}
                </td>
                <td className="px-3 py-1.5 text-right text-psfc-muted">—</td>
                <td className="px-3 py-1.5 text-right text-psfc-muted">s</td>
              </tr>
              <tr className="border-t border-psfc-rule">
                <td className="px-3 py-1.5 text-sm text-psfc-ink">
                  <span className="font-sans">Grating spacing</span>
                </td>
                <td className="px-3 py-1.5 text-right">
                  {active.result.gratingSpacing.toFixed(4)}
                </td>
                <td className="px-3 py-1.5 text-right text-psfc-muted">—</td>
                <td className="px-3 py-1.5 text-right text-psfc-muted">µm</td>
              </tr>
              {PARAM_ORDER.map((p) => {
                const entry = active.result.params[p];
                if (!entry) return null;
                const relErr =
                  entry.value !== 0
                    ? Math.abs(entry.err / entry.value) * 100
                    : Infinity;
                return (
                  <tr key={p} className="border-t border-psfc-rule">
                    <td className="px-3 py-1.5 text-sm text-psfc-ink">
                      <span className="font-sans">{LABELS[p]}</span>
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      {formatValue(entry.value, p)}
                    </td>
                    <td
                      className={cn(
                        "px-3 py-1.5 text-right",
                        relErr > 100 && "text-amber-700",
                        relErr > 1000 && "text-red-700",
                      )}
                    >
                      {formatValue(entry.err, p)}
                      {Number.isFinite(relErr) && (
                        <span className="ml-1 text-[10px] text-psfc-muted">
                          ({relErr.toFixed(1)}%)
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-right text-psfc-muted">
                      {entry.unit}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </SectionCard>
  );
}

function formatValue(v: number, name: FitParamName): string {
  if (!Number.isFinite(v)) return "—";
  if (name === "f") return (v / 1e9).toFixed(4) + " GHz";
  if (name === "tau") return (v * 1e9).toFixed(3) + " ns";
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0)) {
    return v.toExponential(4);
  }
  return v.toFixed(6);
}
