#!/usr/bin/env node
// Copies PyTGS Python source files and config.yaml into public/pytgs/
// so the Pyodide worker can fetch them at runtime.

import { cp, rm, mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, "..");
const repoRoot = resolve(webRoot, "..");
const destRoot = join(webRoot, "public", "pytgs");

const files = [
  "pytgs/__init__.py",
  "pytgs/analyzer.py",
  "pytgs/cli.py",
  "pytgs/fft.py",
  "pytgs/functions.py",
  "pytgs/io.py",
  "pytgs/lorentzian.py",
  "pytgs/paths.py",
  "pytgs/signal_process.py",
  "pytgs/tgs.py",
  "config.yaml",
];

await rm(destRoot, { recursive: true, force: true });
await mkdir(destRoot, { recursive: true });

for (const rel of files) {
  const src = join(repoRoot, rel);
  const dest = join(destRoot, rel);
  if (!existsSync(src)) {
    console.warn(`  skipped (missing): ${rel}`);
    continue;
  }
  await mkdir(dirname(dest), { recursive: true });
  await cp(src, dest);
  console.log(`  copied: ${rel}`);
}

const manifest = { files, generatedAt: new Date().toISOString() };
await writeFile(
  join(destRoot, "manifest.json"),
  JSON.stringify(manifest, null, 2),
);

console.log(`\nCopied ${files.length} files to ${destRoot}`);
