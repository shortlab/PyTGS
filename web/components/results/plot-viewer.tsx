"use client";

import { useMemo, useState } from "react";
import { ImageIcon } from "lucide-react";
import { useAppStore } from "@/lib/app-store";
import type { PlotKind } from "@/lib/config-types";
import { SectionCard } from "@/components/ui/section-card";
import { PlotlyChart } from "./plotly-chart";
import { cn } from "@/lib/utils";
import type { Data, Layout } from "plotly.js";

const TABS: Array<{ kind: PlotKind; label: string }> = [
  { kind: "tgs", label: "TGS fit" },
  { kind: "fft-lorentzian", label: "FFT + Lorentzian" },
];

export function PlotViewer() {
  const { results, activeResultId } = useAppStore();
  const [kind, setKind] = useState<PlotKind>("tgs");

  const active = activeResultId ? results.get(activeResultId) : null;

  return (
    <SectionCard
      title="Fit preview"
      subtitle={
        active
          ? active.displayName
          : "Run a fit to see interactive plots (zoom, pan, hover, export)."
      }
      bodyClassName="p-0"
    >
      {active ? (
        <div>
          <div className="flex gap-1 border-b border-psfc-rule bg-psfc-surface-alt px-3 py-2">
            {TABS.map((t) => (
              <button
                key={t.kind}
                type="button"
                onClick={() => setKind(t.kind)}
                className={cn(
                  "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                  kind === t.kind
                    ? "bg-psfc-red text-white"
                    : "text-psfc-body hover:bg-psfc-rule",
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="p-3">
            {kind === "tgs" ? (
              <TgsPlot result={active.result} />
            ) : (
              <FftLorentzianPlot result={active.result} />
            )}
          </div>
        </div>
      ) : (
        <div className="flex h-[440px] flex-col items-center justify-center gap-2 text-sm text-psfc-muted">
          <ImageIcon size={32} className="opacity-40" />
          <span>No result selected.</span>
        </div>
      )}
    </SectionCard>
  );
}

function TgsPlot({ result }: { result: import("@/lib/config-types").FitResult }) {
  const { data, layout } = useMemo(() => {
    const { timeFull, ampFull, timeFit, yFunctional, yThermal } =
      result.traces.tgs;
    // Convert seconds → nanoseconds, amplitude → mV (match Tkinter's scaling).
    const toNs = (xs: number[]) => xs.map((v) => v * 1e9);
    const toMv = (ys: number[]) => ys.map((v) => v * 1e3);
    const d: Data[] = [
      {
        type: "scattergl",
        name: "Signal",
        x: toNs(timeFull),
        y: toMv(ampFull),
        mode: "lines",
        line: { color: "#1a1a1a", width: 1.25 },
        hovertemplate:
          "%{y:.3f} mV @ %{x:.3f} ns<extra>signal</extra>",
      },
      {
        type: "scattergl",
        name: "Functional fit",
        x: toNs(timeFit),
        y: toMv(yFunctional),
        mode: "lines",
        line: { color: "#0b5ea8", width: 2 },
        hovertemplate:
          "%{y:.3f} mV @ %{x:.3f} ns<extra>functional</extra>",
      },
      {
        type: "scattergl",
        name: "Thermal fit",
        x: toNs(timeFit),
        y: toMv(yThermal),
        mode: "lines",
        line: { color: "#750014", width: 2, dash: "dot" },
        hovertemplate:
          "%{y:.3f} mV @ %{x:.3f} ns<extra>thermal</extra>",
      },
    ];
    const l: Partial<Layout> = {
      xaxis: { title: { text: "Time (ns)" } },
      yaxis: { title: { text: "Signal amplitude (mV)" } },
    };
    return { data: d, layout: l };
  }, [result]);

  return <PlotlyChart data={data} layout={layout} />;
}

function FftLorentzianPlot({
  result,
}: {
  result: import("@/lib/config-types").FitResult;
}) {
  const { data, layout } = useMemo(() => {
    const { frequency, amplitude, curveX, curveY, bounds } =
      result.traces.fftLorentzian;

    const d: Data[] = [
      {
        type: "scattergl",
        name: "FFT",
        x: frequency,
        y: amplitude,
        mode: "lines",
        line: { color: "#1a1a1a", width: 1.25 },
        hovertemplate: "%{y:.3g} @ %{x:.4f} GHz<extra>fft</extra>",
      },
      {
        type: "scattergl",
        name: "Lorentzian fit",
        x: curveX,
        y: curveY,
        mode: "lines",
        line: { color: "#750014", width: 2.5, dash: "dash" },
        hovertemplate:
          "%{y:.3g} @ %{x:.4f} GHz<extra>lorentzian</extra>",
      },
    ];
    const l: Partial<Layout> = {
      xaxis: {
        title: { text: "Frequency (GHz)" },
        range: [0, Math.max(1, bounds[1] + 0.1)],
      },
      yaxis: { title: { text: "Intensity (A.U.)" } },
      shapes: [
        {
          type: "line",
          xref: "x",
          yref: "paper",
          x0: bounds[0],
          x1: bounds[0],
          y0: 0,
          y1: 1,
          line: { color: "#b58900", width: 1.5, dash: "dot" },
        },
        {
          type: "line",
          xref: "x",
          yref: "paper",
          x0: bounds[1],
          x1: bounds[1],
          y0: 0,
          y1: 1,
          line: { color: "#b58900", width: 1.5, dash: "dot" },
        },
      ],
      annotations: [
        {
          x: bounds[0],
          y: 1,
          xref: "x",
          yref: "paper",
          yanchor: "bottom",
          showarrow: false,
          text: "bound",
          font: { size: 10, color: "#b58900" },
        },
        {
          x: bounds[1],
          y: 1,
          xref: "x",
          yref: "paper",
          yanchor: "bottom",
          showarrow: false,
          text: "bound",
          font: { size: 10, color: "#b58900" },
        },
      ],
    };
    return { data: d, layout: l };
  }, [result]);

  return <PlotlyChart data={data} layout={layout} />;
}
