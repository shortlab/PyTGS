"use client";

import { useState } from "react";
import { ArrowDown } from "lucide-react";
import { AppStoreProvider } from "@/lib/app-store";
import { StatusBanner } from "@/components/ui/status-banner";
import { CalibrationSection } from "@/components/sections/calibration-section";
import { ParametersSection } from "@/components/sections/parameters-section";
import { BatchQueueSection } from "@/components/sections/batch-queue-section";
import { ResultsTable } from "@/components/results/results-table";
import { PlotViewer } from "@/components/results/plot-viewer";
import { LogConsole } from "@/components/results/log-console";
import { ConfigEditorModal } from "@/components/config-editor/config-editor-modal";

export default function HomePage() {
  return (
    <AppStoreProvider>
      <HomeContent />
    </AppStoreProvider>
  );
}

function HomeContent() {
  const [configEditorOpen, setConfigEditorOpen] = useState(false);

  return (
    <>
      <Hero />

      <section id="analyzer" className="bg-surface-alt/50 pb-16 pt-10">
        <div className="psfc-container">
          <StatusBanner />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div className="space-y-4">
              <CalibrationSection />
              <ParametersSection
                onOpenEditor={() => setConfigEditorOpen(true)}
              />
              <BatchQueueSection />
            </div>

            <div className="space-y-4">
              <PlotViewer />
              <ResultsTable />
              <LogConsole />
            </div>
          </div>
        </div>
      </section>

      <ConfigEditorModal
        open={configEditorOpen}
        onClose={() => setConfigEditorOpen(false)}
      />
    </>
  );
}

function Hero() {
  return (
    <section className="psfc-hero-bg relative overflow-hidden">
      <div className="psfc-container relative py-14 md:py-20">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] md:items-center">
          <div>
            <span className="psfc-eyebrow text-ink">
              Transient Grating Spectroscopy
            </span>
            <h1 className="mt-4 text-4xl font-bold tracking-tight text-ink md:text-5xl">
              Fit TGS signals in your browser.
              <span className="block text-ink/60">No uploads. No server.</span>
            </h1>
            <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-ink/80">
              Drop POS/NEG file pairs, tune the fit, and inspect the results.
              Everything runs locally — the scientific Python runtime boots once
              and stays warm in your tab.
            </p>

            <div className="mt-7 flex flex-wrap items-center gap-3">
              <a href="#analyzer" className="psfc-btn psfc-btn-primary">
                Start analyzing
                <ArrowDown size={14} />
              </a>
              <a
                href="https://github.com/shortlab/PyTGS/blob/main/README.md"
                target="_blank"
                rel="noopener noreferrer"
                className="psfc-btn"
              >
                Read the docs
              </a>
            </div>
          </div>

          <aside className="hidden md:block">
            <HeroCard />
          </aside>
        </div>
      </div>

    </section>
  );
}

function HeroCard() {
  return (
    <div
      className="relative rounded-2xl border border-ink/10 bg-white/70 p-4 backdrop-blur"
      aria-hidden
    >
      <svg
        viewBox="0 0 320 140"
        className="h-32 w-full"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="sigGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#2a6a94" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#bfd8e8" stopOpacity="0.4" />
          </linearGradient>
        </defs>
        <path
          d="M0,90 C20,30 40,120 60,70 C80,40 100,110 120,65 C140,35 160,95 180,75 C200,60 220,80 240,72 C260,66 280,74 300,70 L320,70"
          fill="none"
          stroke="#0e1117"
          strokeWidth="1.5"
        />
        <path
          d="M0,75 C40,70 80,75 120,72 C160,70 200,73 240,72 C280,71 300,72 320,72"
          fill="none"
          stroke="#1d5f8e"
          strokeWidth="2"
          strokeDasharray="4 3"
        />
        <path
          d="M0,90 C20,30 40,120 60,70 C80,40 100,110 120,65 C140,35 160,95 180,75 C200,60 220,80 240,72 C260,66 280,74 300,70 L320,70 L320,140 L0,140 Z"
          fill="url(#sigGrad)"
          opacity="0.4"
        />
      </svg>
    </div>
  );
}
