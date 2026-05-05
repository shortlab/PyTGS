import { ArrowUpRight } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-auto bg-dark text-white/80">
      <div className="psfc-container grid grid-cols-1 gap-10 py-12 md:grid-cols-4">
        <div className="md:col-span-2">
          <div className="flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden>
              <rect x="2" y="10" width="2.4" height="7" rx="1" fill="#bfd8e8" />
              <rect x="6" y="5" width="2.4" height="12" rx="1" fill="#1d5f8e" />
              <rect x="10" y="2" width="2.4" height="15" rx="1" fill="#ffffff" />
              <rect x="14" y="7" width="2.4" height="10" rx="1" fill="#bfd8e8" />
            </svg>
            <span className="text-sm font-semibold tracking-tight text-white">
              PyTGS
            </span>
          </div>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-white/70">
            An open-source analyzer for Transient Grating Spectroscopy signals.
            Built at the MIT Plasma Science and Fusion Center. Runs entirely in
            your browser via Pyodide — your data never leaves the machine.
          </p>
        </div>

        <FooterCol
          heading="Project"
          links={[
            {
              label: "Source on GitHub",
              href: "https://github.com/shortlab/PyTGS",
              external: true,
            },
            {
              label: "Documentation",
              href: "https://github.com/shortlab/PyTGS/blob/main/README.md",
              external: true,
            },
            {
              label: "Report an issue",
              href: "https://github.com/shortlab/PyTGS/issues",
              external: true,
            },
          ]}
        />
        <FooterCol
          heading="MIT"
          links={[
            {
              label: "Plasma Science & Fusion Center",
              href: "https://www.psfc.mit.edu/",
              external: true,
            },
            {
              label: "Short Lab",
              href: "https://short.mit.edu/",
              external: true,
            },
            { label: "MIT", href: "https://web.mit.edu/", external: true },
          ]}
        />
      </div>
      <div className="border-t border-white/10">
        <div className="psfc-container flex flex-col items-start justify-between gap-2 py-5 text-xs text-white/55 md:flex-row md:items-center">
          <span>
            © {new Date().getFullYear()} Massachusetts Institute of Technology
          </span>
          <span className="inline-flex items-center gap-3">
            <span>v0.2.0 · runs offline after first load</span>
            <a
              href="https://github.com/shortlab/PyTGS"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-full border border-white/15 px-2.5 py-1 text-white/70 transition-colors hover:border-white/40 hover:text-white"
            >
              <svg
                width="11"
                height="11"
                viewBox="0 0 16 16"
                fill="currentColor"
                aria-hidden
              >
                <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
              </svg>
              shortlab/PyTGS
            </a>
          </span>
        </div>
      </div>
    </footer>
  );
}

function FooterCol({
  heading,
  links,
}: {
  heading: string;
  links: { label: string; href: string; external?: boolean }[];
}) {
  return (
    <div>
      <h4 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-white/90">
        {heading}
      </h4>
      <ul className="space-y-2 text-sm">
        {links.map((l) => (
          <li key={l.href}>
            <a
              href={l.href}
              target={l.external ? "_blank" : undefined}
              rel={l.external ? "noopener noreferrer" : undefined}
              className="group inline-flex items-center gap-1 text-white/70 transition-colors hover:text-white"
            >
              {l.label}
              {l.external && (
                <ArrowUpRight
                  size={11}
                  className="opacity-50 transition-opacity group-hover:opacity-100"
                />
              )}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
