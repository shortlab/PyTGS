import Link from "next/link";
import { BookOpen, ArrowUpRight } from "lucide-react";

export function Header() {
  return (
    <header className="sticky top-0 z-30 bg-dark text-white backdrop-blur">
      <div className="psfc-container flex h-16 items-center justify-between gap-4">
        <Link
          href="/"
          className="group flex items-center gap-3"
          aria-label="PyTGS home"
        >
          <BrandMark />
          <div className="flex flex-col leading-tight">
            <span className="text-[15px] font-semibold tracking-tight text-white transition-colors group-hover:text-brand">
              PyTGS
            </span>
            <span className="hidden text-[11px] font-medium uppercase tracking-[0.14em] text-white/55 sm:inline">
              Plasma Science · Fusion Center
            </span>
          </div>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          <a
            href="https://github.com/shortlab/PyTGS/blob/main/README.md"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] text-white/80 transition-colors hover:bg-white/10 hover:text-white"
          >
            <BookOpen size={14} />
            Docs
          </a>
          <a
            href="https://www.psfc.mit.edu/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-[13px] text-white/80 transition-colors hover:bg-white/10 hover:text-white"
          >
            MIT PSFC
            <ArrowUpRight size={12} className="opacity-70" />
          </a>
          <span
            aria-hidden
            className="mx-1 h-4 w-px bg-white/15"
          />
          <a
            href="https://github.com/shortlab/PyTGS"
            target="_blank"
            rel="noopener noreferrer"
            className="psfc-btn psfc-btn-on-dark"
          >
            <GithubGlyph />
            GitHub
          </a>
        </nav>

        <a
          href="https://github.com/shortlab/PyTGS"
          target="_blank"
          rel="noopener noreferrer"
          className="psfc-btn psfc-btn-on-dark md:hidden"
          aria-label="GitHub"
        >
          <GithubGlyph />
        </a>
      </div>
      <div className="psfc-divider-thick" />
    </header>
  );
}

function GithubGlyph() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 16 16"
      fill="currentColor"
      aria-hidden
    >
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function BrandMark() {
  return (
    <span
      aria-hidden
      className="flex h-9 w-9 items-center justify-center rounded-full bg-white/5 ring-1 ring-white/10"
    >
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <rect x="2" y="10" width="2.4" height="7" rx="1" fill="#bfd8e8" />
        <rect x="6" y="5" width="2.4" height="12" rx="1" fill="#1d5f8e" />
        <rect x="10" y="2" width="2.4" height="15" rx="1" fill="#ffffff" />
        <rect x="14" y="7" width="2.4" height="10" rx="1" fill="#bfd8e8" />
      </svg>
    </span>
  );
}
