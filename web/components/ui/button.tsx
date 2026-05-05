import type { ButtonHTMLAttributes } from "react";
import { forwardRef } from "react";
import { cn } from "@/lib/utils";

type Variant = "default" | "primary" | "ghost" | "danger" | "dark";
type Size = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { variant = "default", size = "md", className, type = "button", ...rest },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={cn(
          "psfc-btn",
          variant === "primary" && "psfc-btn-primary",
          variant === "ghost" && "psfc-btn-ghost",
          variant === "dark" && "psfc-btn-dark",
          variant === "danger" &&
            "border-rose-200 bg-rose-50 text-rose-700 hover:border-rose-300 hover:bg-rose-100",
          size === "sm" && "px-3 py-1 text-xs",
          className,
        )}
        {...rest}
      />
    );
  },
);
