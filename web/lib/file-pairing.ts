// Pair uploaded files into POS/NEG pairs using the PyTGS filename convention:
//   {prefix}-{study}-{POS|NEG}-{index}.txt

export interface FilePair {
  id: string;
  study: string;
  index: number;
  prefix: string;
  posFile: File;
  negFile: File;
  displayName: string;
}

export interface PairingReport {
  pairs: FilePair[];
  unpaired: Array<{ file: File; reason: string }>;
}

const RX = /^(.+)-([\w.]+)-(POS|NEG)-(\d+)\.txt$/i;

interface Parsed {
  file: File;
  prefix: string;
  study: string;
  sign: "POS" | "NEG";
  index: number;
}

export function pairFiles(files: File[]): PairingReport {
  const parsed: Parsed[] = [];
  const unpaired: Array<{ file: File; reason: string }> = [];

  for (const file of files) {
    const m = RX.exec(file.name);
    if (!m) {
      unpaired.push({ file, reason: "Filename does not match {prefix}-{study}-{POS|NEG}-{index}.txt" });
      continue;
    }
    parsed.push({
      file,
      prefix: m[1],
      study: m[2],
      sign: m[3].toUpperCase() as "POS" | "NEG",
      index: parseInt(m[4], 10),
    });
  }

  const byKey = new Map<string, { pos?: Parsed; neg?: Parsed }>();
  for (const p of parsed) {
    const key = `${p.prefix}|${p.study}|${p.index}`;
    const slot = byKey.get(key) ?? {};
    if (p.sign === "POS") slot.pos = p;
    else slot.neg = p;
    byKey.set(key, slot);
  }

  const pairs: FilePair[] = [];
  for (const [, slot] of byKey) {
    if (slot.pos && slot.neg) {
      pairs.push({
        id: `${slot.pos.prefix}|${slot.pos.study}|${slot.pos.index}`,
        study: slot.pos.study,
        index: slot.pos.index,
        prefix: slot.pos.prefix,
        posFile: slot.pos.file,
        negFile: slot.neg.file,
        displayName: `${slot.pos.prefix}-${slot.pos.study}-${slot.pos.index}`,
      });
    } else if (slot.pos) {
      unpaired.push({ file: slot.pos.file, reason: "No matching NEG file" });
    } else if (slot.neg) {
      unpaired.push({ file: slot.neg.file, reason: "No matching POS file" });
    }
  }

  pairs.sort((a, b) => {
    if (a.study !== b.study) return a.study.localeCompare(b.study);
    return a.index - b.index;
  });

  return { pairs, unpaired };
}
