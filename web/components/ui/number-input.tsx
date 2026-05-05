import type { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface NumberInputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "value"> {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}

export function NumberInput({
  value,
  onChange,
  className,
  min,
  max,
  step = "any" as unknown as number,
  ...rest
}: NumberInputProps) {
  return (
    <input
      type="number"
      className={cn("psfc-input", className)}
      value={Number.isFinite(value) ? value : ""}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const n = parseFloat(e.target.value);
        onChange(Number.isFinite(n) ? n : 0);
      }}
      {...rest}
    />
  );
}
