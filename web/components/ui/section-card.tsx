import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SectionCardProps {
  step?: number;
  title: string;
  subtitle?: string;
  headerRight?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

export function SectionCard({
  step,
  title,
  subtitle,
  headerRight,
  children,
  className,
  bodyClassName,
}: SectionCardProps) {
  return (
    <section className={cn("psfc-card", className)}>
      <header className="flex items-start justify-between gap-4 border-b border-rule px-5 py-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {step != null && (
              <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-ink text-[11px] font-bold text-white">
                {step}
              </span>
            )}
            <span className="psfc-section-title">{title}</span>
          </div>
          {subtitle && (
            <p className="mt-1.5 text-[13px] leading-snug text-muted">
              {subtitle}
            </p>
          )}
        </div>
        {headerRight && <div className="shrink-0">{headerRight}</div>}
      </header>
      <div className={cn("px-5 py-4", bodyClassName)}>{children}</div>
    </section>
  );
}
