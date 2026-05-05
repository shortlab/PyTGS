"use client";

import dynamic from "next/dynamic";
import type { ComponentType, CSSProperties } from "react";
import type { Data, Layout, Config } from "plotly.js";

// Load Plotly on the client only. We use react-plotly.js's factory with the
// pre-bundled min dist so we don't pull the full ~11 MB source tree through
// webpack. The factory is also what skips prerender (no window on the server).
const Plot = dynamic(
  () =>
    (async () => {
      const [{ default: createPlotlyComponent }, { default: Plotly }] =
        await Promise.all([
          import("react-plotly.js/factory"),
          import("plotly.js-dist-min"),
        ]);
      return {
        default: createPlotlyComponent(
          Plotly as unknown as Parameters<typeof createPlotlyComponent>[0],
        ) as unknown as ComponentType<PlotProps>,
      };
    })(),
  { ssr: false, loading: () => <ChartSkeleton /> },
);

interface PlotProps {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  style?: CSSProperties;
  useResizeHandler?: boolean;
  className?: string;
}

export interface PlotlyChartProps {
  data: Data[];
  layout: Partial<Layout>;
  height?: number;
  className?: string;
}

const BASE_CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  toImageButtonOptions: {
    format: "png",
    filename: "pytgs-plot",
    height: 600,
    width: 1100,
    scale: 2,
  },
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

const BASE_LAYOUT: Partial<Layout> = {
  autosize: true,
  margin: { l: 64, r: 24, t: 16, b: 48 },
  font: {
    family:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Roboto, sans-serif',
    size: 12,
    color: "#1a1a1a",
  },
  plot_bgcolor: "#ffffff",
  paper_bgcolor: "#ffffff",
  hoverlabel: {
    bgcolor: "#1a1a1a",
    bordercolor: "#1a1a1a",
    font: { color: "#ffffff", size: 12 },
  },
  legend: {
    orientation: "h",
    yanchor: "bottom",
    y: 1.02,
    xanchor: "right",
    x: 1,
    bgcolor: "rgba(255,255,255,0)",
  },
  xaxis: {
    gridcolor: "#eeeeef",
    zerolinecolor: "#d1d5db",
    linecolor: "#d1d5db",
    ticks: "outside",
    tickcolor: "#9ca3af",
    mirror: false,
    showline: true,
  },
  yaxis: {
    gridcolor: "#eeeeef",
    zerolinecolor: "#d1d5db",
    linecolor: "#d1d5db",
    ticks: "outside",
    tickcolor: "#9ca3af",
    mirror: false,
    showline: true,
  },
};

export function PlotlyChart({
  data,
  layout,
  height = 440,
  className,
}: PlotlyChartProps) {
  const merged: Partial<Layout> = {
    ...BASE_LAYOUT,
    ...layout,
    xaxis: { ...BASE_LAYOUT.xaxis, ...layout.xaxis },
    yaxis: { ...BASE_LAYOUT.yaxis, ...layout.yaxis },
    font: { ...BASE_LAYOUT.font, ...layout.font },
    legend: { ...BASE_LAYOUT.legend, ...layout.legend },
  };
  return (
    <Plot
      data={data}
      layout={merged}
      config={BASE_CONFIG}
      useResizeHandler
      className={className}
      style={{ width: "100%", height }}
    />
  );
}

function ChartSkeleton() {
  return (
    <div className="flex h-[440px] w-full animate-pulse items-center justify-center bg-psfc-surface-alt text-sm text-psfc-muted">
      Loading plot…
    </div>
  );
}
