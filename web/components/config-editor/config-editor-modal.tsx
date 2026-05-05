"use client";

import { useEffect, useState } from "react";
import { X, RefreshCcw, Check } from "lucide-react";
import { useAppStore } from "@/lib/app-store";
import { DEFAULT_CONFIG } from "@/lib/default-config";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { NumberInput } from "@/components/ui/number-input";
import type {
  FftAnalysisType,
  HeterodyneMode,
  PyTGSConfig,
} from "@/lib/config-types";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function ConfigEditorModal({ open, onClose }: Props) {
  const { config, setConfig } = useAppStore();
  const [draft, setDraft] = useState<PyTGSConfig>(config);

  useEffect(() => {
    if (open) setDraft(structuredClone(config));
  }, [open, config]);

  if (!open) return null;

  function update<K extends keyof PyTGSConfig>(
    key: K,
    patch: Partial<PyTGSConfig[K]>,
  ) {
    setDraft((prev) => ({
      ...prev,
      [key]: { ...(prev[key] as object), ...patch } as PyTGSConfig[K],
    }));
  }

  function commit() {
    setConfig(draft);
    onClose();
  }

  function reset() {
    setDraft(structuredClone(DEFAULT_CONFIG));
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        className="scrollbar-thin max-h-[90vh] w-full max-w-3xl overflow-auto rounded-lg bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-psfc-rule bg-white px-6 py-4">
          <div>
            <h2 className="text-lg font-semibold text-psfc-ink">
              Edit configuration
            </h2>
            <p className="text-xs text-psfc-muted">
              Mirrors config.yaml. Takes effect for the next fit.
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" size="sm" onClick={reset}>
              <RefreshCcw size={14} />
              Reset defaults
            </Button>
            <button
              onClick={onClose}
              className="rounded p-1 text-psfc-muted hover:bg-psfc-surface-alt"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="space-y-6 px-6 py-5">
          <Group title="Signal processing">
            <Field label="Heterodyne mode">
              <select
                className="psfc-input"
                value={draft.signal_process.heterodyne}
                onChange={(e) =>
                  update("signal_process", {
                    heterodyne: e.target.value as HeterodyneMode,
                  })
                }
              >
                <option value="di-homodyne">di-homodyne</option>
                <option value="mono-homodyne">mono-homodyne</option>
              </select>
            </Field>
            <Field label="Null point (1-4)">
              <NumberInput
                value={draft.signal_process.null_point}
                onChange={(v) =>
                  update("signal_process", {
                    null_point: Math.min(4, Math.max(1, Math.round(v))) as
                      | 1
                      | 2
                      | 3
                      | 4,
                  })
                }
                min={1}
                max={4}
                step={1}
              />
            </Field>
            <Field label="Initial samples">
              <NumberInput
                value={draft.signal_process.initial_samples}
                onChange={(v) =>
                  update("signal_process", {
                    initial_samples: Math.max(1, Math.round(v)),
                  })
                }
                min={1}
                step={1}
              />
            </Field>
            <Field label="Baseline correction" fullWidth>
              <Checkbox
                label="Enable baseline correction"
                hint="subtract a reference POS/NEG pair"
                checked={draft.signal_process.baseline_correction.enabled}
                onChange={(e) =>
                  update("signal_process", {
                    baseline_correction: {
                      ...draft.signal_process.baseline_correction,
                      enabled: e.target.checked,
                    },
                  })
                }
              />
            </Field>
          </Group>

          <Group title="FFT analysis">
            <Field label="Analysis type">
              <select
                className="psfc-input"
                value={draft.fft.analysis_type}
                onChange={(e) =>
                  update("fft", {
                    analysis_type: e.target.value as FftAnalysisType,
                  })
                }
              >
                <option value="fft">FFT</option>
                <option value="psd">PSD</option>
              </select>
            </Field>
            <Field label="Signal proportion (0–1)">
              <NumberInput
                value={draft.fft.signal_proportion}
                onChange={(v) => update("fft", { signal_proportion: v })}
                min={0}
                max={1}
                step={0.01}
              />
            </Field>
          </Group>

          <Group title="Lorentzian fitting">
            <Field label="Signal proportion (0–1)">
              <NumberInput
                value={draft.lorentzian.signal_proportion}
                onChange={(v) =>
                  update("lorentzian", { signal_proportion: v })
                }
                min={0}
                max={1}
                step={0.01}
              />
            </Field>
            <Field label="Freq. lower bound (GHz)">
              <NumberInput
                value={draft.lorentzian.frequency_bounds[0]}
                onChange={(v) =>
                  update("lorentzian", {
                    frequency_bounds: [v, draft.lorentzian.frequency_bounds[1]],
                  })
                }
                step={0.01}
              />
            </Field>
            <Field label="Freq. upper bound (GHz)">
              <NumberInput
                value={draft.lorentzian.frequency_bounds[1]}
                onChange={(v) =>
                  update("lorentzian", {
                    frequency_bounds: [draft.lorentzian.frequency_bounds[0], v],
                  })
                }
                step={0.01}
              />
            </Field>
            <Field label="DC filter start (bin)">
              <NumberInput
                value={draft.lorentzian.dc_filter_range[0]}
                onChange={(v) =>
                  update("lorentzian", {
                    dc_filter_range: [
                      Math.max(0, Math.round(v)),
                      draft.lorentzian.dc_filter_range[1],
                    ],
                  })
                }
                min={0}
                step={1}
              />
            </Field>
            <Field label="DC filter end (bin)">
              <NumberInput
                value={draft.lorentzian.dc_filter_range[1]}
                onChange={(v) =>
                  update("lorentzian", {
                    dc_filter_range: [
                      draft.lorentzian.dc_filter_range[0],
                      Math.max(0, Math.round(v)),
                    ],
                  })
                }
                min={0}
                step={100}
              />
            </Field>
            <Field label="Peak model" fullWidth>
              <div className="space-y-2">
                <Checkbox
                  label="Bimodal fit"
                  hint="two SAW peaks"
                  checked={draft.lorentzian.bimodal_fit}
                  onChange={(e) =>
                    update("lorentzian", { bimodal_fit: e.target.checked })
                  }
                />
                <Checkbox
                  label="Skewed super-Lorentzian"
                  hint="asymmetric line shape"
                  checked={draft.lorentzian.use_skewed_super_lorentzian}
                  onChange={(e) =>
                    update("lorentzian", {
                      use_skewed_super_lorentzian: e.target.checked,
                    })
                  }
                />
              </div>
            </Field>
          </Group>

          <Group title="TGS fit">
            <Field label="Grating spacing (µm)">
              <NumberInput
                value={draft.tgs.grating_spacing}
                onChange={(v) => update("tgs", { grating_spacing: v })}
                step={0.0001}
              />
            </Field>
            <Field label="Signal proportion (0–1)">
              <NumberInput
                value={draft.tgs.signal_proportion}
                onChange={(v) => update("tgs", { signal_proportion: v })}
                min={0}
                max={1}
                step={0.01}
              />
            </Field>
            <Field label="Max curve_fit iterations">
              <NumberInput
                value={draft.tgs.maxfev}
                onChange={(v) =>
                  update("tgs", { maxfev: Math.max(1, Math.round(v)) })
                }
                step={1000}
              />
            </Field>
          </Group>

        </div>

        <div className="sticky bottom-0 flex justify-end gap-2 border-t border-psfc-rule bg-white px-6 py-3">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={commit}>
            <Check size={14} />
            Save changes
          </Button>
        </div>
      </div>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-psfc-red">
        {title}
      </h3>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">{children}</div>
    </section>
  );
}

function Field({
  label,
  children,
  fullWidth,
}: {
  label: string;
  children: React.ReactNode;
  fullWidth?: boolean;
}) {
  return (
    <div className={fullWidth ? "col-span-2" : ""}>
      <span className="psfc-label">{label}</span>
      {children}
    </div>
  );
}
