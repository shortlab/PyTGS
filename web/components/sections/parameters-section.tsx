"use client";

import { Settings2 } from "lucide-react";
import { useAppStore } from "@/lib/app-store";
import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/ui/section-card";
import { Checkbox } from "@/components/ui/checkbox";
import { NumberInput } from "@/components/ui/number-input";
import { FilePairPicker } from "@/components/ui/file-pair-picker";

interface ParametersSectionProps {
  onOpenEditor: () => void;
}

export function ParametersSection({ onOpenEditor }: ParametersSectionProps) {
  const {
    config,
    updateConfig,
    baselineFiles,
    setBaselineFiles,
    isRunning,
  } = useAppStore();

  return (
    <SectionCard
      step={2}
      title="Global parameters"
      subtitle="Quick-access knobs. Use the full editor for advanced options."
      headerRight={
        <Button variant="ghost" size="sm" onClick={onOpenEditor}>
          <Settings2 size={14} />
          Edit all…
        </Button>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <span className="psfc-label">Null point (1-4)</span>
            <NumberInput
              value={config.signal_process.null_point}
              onChange={(v) =>
                updateConfig((c) => {
                  c.signal_process.null_point = Math.min(
                    4,
                    Math.max(1, Math.round(v)),
                  ) as 1 | 2 | 3 | 4;
                  return c;
                })
              }
              min={1}
              max={4}
              step={1}
              disabled={isRunning}
            />
          </div>
          <div>
            <span className="psfc-label">Frequency bounds (GHz)</span>
            <div className="flex items-center gap-2">
              <NumberInput
                value={config.lorentzian.frequency_bounds[0]}
                onChange={(v) =>
                  updateConfig((c) => {
                    c.lorentzian.frequency_bounds[0] = v;
                    return c;
                  })
                }
                step={0.01}
                disabled={isRunning}
              />
              <span className="text-psfc-muted">–</span>
              <NumberInput
                value={config.lorentzian.frequency_bounds[1]}
                onChange={(v) =>
                  updateConfig((c) => {
                    c.lorentzian.frequency_bounds[1] = v;
                    return c;
                  })
                }
                step={0.01}
                disabled={isRunning}
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Checkbox
            label="Two SAW peaks"
            hint="bimodal Lorentzian fit"
            checked={config.lorentzian.bimodal_fit}
            onChange={(e) =>
              updateConfig((c) => {
                c.lorentzian.bimodal_fit = e.target.checked;
                return c;
              })
            }
            disabled={isRunning}
          />
          <Checkbox
            label="Skewed super-Lorentzian"
            hint="asymmetric peak"
            checked={config.lorentzian.use_skewed_super_lorentzian}
            onChange={(e) =>
              updateConfig((c) => {
                c.lorentzian.use_skewed_super_lorentzian = e.target.checked;
                return c;
              })
            }
            disabled={isRunning}
          />
          <Checkbox
            label="PSD analysis"
            hint="use power spectral density instead of FFT"
            checked={config.fft.analysis_type === "psd"}
            onChange={(e) =>
              updateConfig((c) => {
                c.fft.analysis_type = e.target.checked ? "psd" : "fft";
                return c;
              })
            }
            disabled={isRunning}
          />
          <Checkbox
            label="Baseline correction"
            hint="subtract reference signal"
            checked={config.signal_process.baseline_correction.enabled}
            onChange={(e) =>
              updateConfig((c) => {
                c.signal_process.baseline_correction.enabled = e.target.checked;
                return c;
              })
            }
            disabled={isRunning}
          />
        </div>

        {config.signal_process.baseline_correction.enabled && (
          <FilePairPicker
            label="Baseline reference files"
            pos={baselineFiles.pos}
            neg={baselineFiles.neg}
            onChange={setBaselineFiles}
            disabled={isRunning}
            helper="POS/NEG files recorded with the pump off for baseline subtraction."
          />
        )}
      </div>
    </SectionCard>
  );
}
