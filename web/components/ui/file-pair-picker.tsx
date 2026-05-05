"use client";

import { useRef } from "react";
import { Upload, CheckCircle2, AlertCircle, X } from "lucide-react";
import { Button } from "./button";
import { cn } from "@/lib/utils";

interface FilePairPickerProps {
  label: string;
  pos: File | null;
  neg: File | null;
  onChange: (pair: { pos: File | null; neg: File | null }) => void;
  disabled?: boolean;
  helper?: string;
}

// Accepts up to 2 files; auto-assigns POS/NEG based on filename.
export function FilePairPicker({
  label,
  pos,
  neg,
  onChange,
  disabled,
  helper,
}: FilePairPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    let nextPos: File | null = pos;
    let nextNeg: File | null = neg;
    for (const f of Array.from(files)) {
      const name = f.name.toUpperCase();
      if (name.includes("-POS-")) nextPos = f;
      else if (name.includes("-NEG-")) nextNeg = f;
      else if (!nextPos) nextPos = f;
      else if (!nextNeg) nextNeg = f;
    }
    onChange({ pos: nextPos, neg: nextNeg });
  }

  const hasBoth = pos && neg;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="psfc-label mb-0">{label}</span>
        {(pos || neg) && !disabled && (
          <button
            type="button"
            onClick={() => onChange({ pos: null, neg: null })}
            className="text-xs text-psfc-muted hover:text-psfc-red"
          >
            clear
          </button>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".txt"
        multiple
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <div
        className={cn(
          "rounded-xl border border-dashed p-3 transition-colors",
          hasBoth
            ? "border-emerald-300 bg-emerald-50/50"
            : pos || neg
              ? "border-amber-300 bg-amber-50/40"
              : "border-rule bg-surface-alt",
        )}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 flex-1 space-y-1">
            <FilePairRow
              tag="POS"
              file={pos}
              onRemove={() => onChange({ pos: null, neg })}
              disabled={disabled}
            />
            <FilePairRow
              tag="NEG"
              file={neg}
              onRemove={() => onChange({ pos, neg: null })}
              disabled={disabled}
            />
          </div>
          <Button
            variant="default"
            size="sm"
            disabled={disabled}
            onClick={() => inputRef.current?.click()}
          >
            <Upload size={14} />
            Choose files
          </Button>
        </div>
        {helper && !pos && !neg && (
          <p className="mt-2 text-xs text-psfc-muted">{helper}</p>
        )}
      </div>
    </div>
  );
}

function FilePairRow({
  tag,
  file,
  onRemove,
  disabled,
}: {
  tag: "POS" | "NEG";
  file: File | null;
  onRemove: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={cn(
          "inline-flex w-11 shrink-0 items-center justify-center rounded-full px-1.5 py-0.5 font-mono text-[10px] font-bold",
          file ? "bg-ink text-white" : "bg-rule text-muted",
        )}
      >
        {tag}
      </span>
      {file ? (
        <>
          <CheckCircle2 size={14} className="shrink-0 text-green-600" />
          <span className="min-w-0 truncate font-mono text-xs text-psfc-body">
            {file.name}
          </span>
          {!disabled && (
            <button
              type="button"
              onClick={onRemove}
              className="ml-auto text-psfc-muted hover:text-psfc-red"
              aria-label={`Remove ${tag} file`}
            >
              <X size={14} />
            </button>
          )}
        </>
      ) : (
        <>
          <AlertCircle size={14} className="shrink-0 text-amber-500" />
          <span className="text-xs text-psfc-muted">
            No {tag} file selected
          </span>
        </>
      )}
    </div>
  );
}
