"use client";

import { useState } from "react";
import { Play } from "lucide-react";
import { useAppStore } from "@/lib/app-store";
import { DEFAULT_CALIB_SOUND_SPEED } from "@/lib/default-config";
import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/ui/section-card";
import { FilePairPicker } from "@/components/ui/file-pair-picker";
import { NumberInput } from "@/components/ui/number-input";

export function CalibrationSection() {
  const {
    config,
    updateConfig,
    calibrationFiles,
    setCalibrationFiles,
    calibrationResult,
    setCalibrationResult,
    pool,
    initPool,
    appendLog,
    isRunning,
  } = useAppStore();

  const [soundSpeed, setSoundSpeed] = useState(DEFAULT_CALIB_SOUND_SPEED);
  const [busy, setBusy] = useState(false);

  const hasPair = calibrationFiles.pos && calibrationFiles.neg;

  async function handleRun() {
    if (!calibrationFiles.pos || !calibrationFiles.neg) return;
    setBusy(true);
    appendLog({
      level: "info",
      message: `Running calibration on ${calibrationFiles.pos.name} / ${calibrationFiles.neg.name}…`,
      timestamp: Date.now(),
    });
    try {
      await initPool();
      const posBytes = await calibrationFiles.pos.arrayBuffer();
      const negBytes = await calibrationFiles.neg.arrayBuffer();
      const result = await pool.calibrate({
        posBytes,
        negBytes,
        config,
        soundSpeed,
      });
      setCalibrationResult(result);
      updateConfig((c) => {
        c.tgs.grating_spacing = result.gratingSpacing;
        return c;
      });
      appendLog({
        level: "info",
        message: `Calibration complete: grating spacing = ${result.gratingSpacing.toFixed(4)} µm (SAW = ${(result.frequency / 1e9).toFixed(4)} GHz)`,
        timestamp: Date.now(),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      appendLog({
        level: "error",
        message: `Calibration failed: ${msg}`,
        timestamp: Date.now(),
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <SectionCard
      step={1}
      title="Calibration"
      subtitle="Derive grating spacing from a reference signal with a known SAW velocity."
    >
      <div className="space-y-4">
        <FilePairPicker
          label="Calibration files"
          pos={calibrationFiles.pos}
          neg={calibrationFiles.neg}
          onChange={setCalibrationFiles}
          disabled={busy || isRunning}
          helper="Select matching POS and NEG reference files (drag both in one go)."
        />

        <div className="grid grid-cols-2 gap-4">
          <div>
            <span className="psfc-label">SAW sound speed (m/s)</span>
            <NumberInput
              value={soundSpeed}
              onChange={setSoundSpeed}
              step={0.1}
              disabled={busy || isRunning}
            />
          </div>
          <div>
            <span className="psfc-label">Grating spacing (µm)</span>
            <NumberInput
              value={config.tgs.grating_spacing}
              onChange={(v) =>
                updateConfig((c) => {
                  c.tgs.grating_spacing = v;
                  return c;
                })
              }
              step={0.0001}
              disabled={busy || isRunning}
            />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-xs text-psfc-muted">
            {calibrationResult ? (
              <span>
                SAW ={" "}
                <span className="font-mono text-psfc-ink">
                  {(calibrationResult.frequency / 1e9).toFixed(4)} GHz
                </span>
                {" · "}
                SNR ={" "}
                <span className="font-mono text-psfc-ink">
                  {calibrationResult.snr.toFixed(1)}
                </span>
              </span>
            ) : (
              <span>Uses your SAW velocity to solve λ = v / f.</span>
            )}
          </div>
          <Button
            variant="primary"
            onClick={handleRun}
            disabled={!hasPair || busy || isRunning}
          >
            <Play size={14} />
            {busy ? "Running…" : "Run calibration"}
          </Button>
        </div>
      </div>
    </SectionCard>
  );
}
