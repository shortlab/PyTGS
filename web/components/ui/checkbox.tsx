import type { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface CheckboxProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  label: string;
  hint?: string;
  checkboxClassName?: string;
}

export function Checkbox({
  label,
  hint,
  className,
  checkboxClassName,
  ...rest
}: CheckboxProps) {
  return (
    <label
      className={cn(
        "flex cursor-pointer select-none items-start gap-2 text-sm",
        rest.disabled && "cursor-not-allowed opacity-50",
        className,
      )}
    >
      <input
        type="checkbox"
        className={cn(
          "mt-0.5 h-4 w-4 accent-psfc-red",
          checkboxClassName,
        )}
        {...rest}
      />
      <span className="leading-tight">
        <span className="font-medium text-psfc-ink">{label}</span>
        {hint && (
          <span className="ml-1 text-xs text-psfc-muted">— {hint}</span>
        )}
      </span>
    </label>
  );
}
