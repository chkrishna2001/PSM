export type SegmentSplitReason = "single" | "markdown_header" | "numbered_block" | "paragraph" | "hard_max";

export interface SegmentOptions {
  maxChunkTokens?: number;
  minChunkTokens?: number;
}

export interface TextSegment {
  index: number;
  text: string;
  estimatedTokens: number;
  splitReason: SegmentSplitReason;
}

const DEFAULT_MAX_CHUNK_TOKENS = 1200;
const DEFAULT_MIN_CHUNK_TOKENS = 200;

export function estimateTextTokens(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return Math.max(1, Math.ceil(trimmed.length / 4));
}

export function segmentLlmResponse(text: string, options: SegmentOptions = {}): TextSegment[] {
  const maxChunkTokens = options.maxChunkTokens ?? DEFAULT_MAX_CHUNK_TOKENS;
  const minChunkTokens = options.minChunkTokens ?? DEFAULT_MIN_CHUNK_TOKENS;
  const trimmed = text.trim();
  if (!trimmed) return [];

  if (estimateTextTokens(trimmed) <= maxChunkTokens) {
    return [makeSegment(0, trimmed, "single")];
  }

  const headerSections = splitByMarkdownHeaders(trimmed);
  const units: Array<{ text: string; reason: SegmentSplitReason }> = [];
  for (const section of headerSections) {
    const reason: SegmentSplitReason = headerSections.length > 1 ? "markdown_header" : "paragraph";
    for (const unit of splitPreservingNumberedBlocks(section)) {
      const unitReason = isNumberedWorkflowBlock(unit) ? "numbered_block" : reason;
      if (estimateTextTokens(unit) <= maxChunkTokens) {
        units.push({ text: unit, reason: unitReason });
      } else {
        for (const piece of hardSplitByTokens(unit, maxChunkTokens)) {
          units.push({ text: piece, reason: "hard_max" });
        }
      }
    }
  }

  const merged = mergeSmallSegments(units, minChunkTokens, maxChunkTokens);
  return merged.map((unit, index) => makeSegment(index, unit.text, unit.reason));
}

export function chunkSourceId(baseSourceId: string, chunkIndex: number): string {
  return `${baseSourceId}:chunk-${chunkIndex}`;
}

export function baseSourceId(sourceId: string): string {
  return sourceId.replace(/:chunk-\d+$/, "");
}

export function normalizeContentKey(content: string): string {
  return content.trim().toLowerCase().replace(/\s+/g, " ");
}

function makeSegment(index: number, text: string, splitReason: SegmentSplitReason): TextSegment {
  return {
    index,
    text,
    estimatedTokens: estimateTextTokens(text),
    splitReason
  };
}

function splitByMarkdownHeaders(text: string): string[] {
  const lines = text.split("\n");
  const sections: string[] = [];
  let current: string[] = [];

  const flush = () => {
    const joined = current.join("\n").trim();
    if (joined) sections.push(joined);
    current = [];
  };

  for (const line of lines) {
    if (/^#{1,3}\s+/.test(line) && current.length > 0) {
      flush();
    }
    current.push(line);
  }
  flush();
  return sections.length > 0 ? sections : [text.trim()];
}

function splitPreservingNumberedBlocks(text: string): string[] {
  const lines = text.split("\n");
  const units: string[] = [];
  let buffer: string[] = [];
  let numberedCount = 0;

  const flush = () => {
    const joined = buffer.join("\n").trim();
    if (joined) units.push(joined);
    buffer = [];
    numberedCount = 0;
  };

  for (const line of lines) {
    const isNumbered = /^\s*\d+\.\s+/.test(line);
    const isHeader = /^#{1,3}\s+/.test(line);

    if (isHeader) {
      flush();
      buffer.push(line);
      continue;
    }

    if (isNumbered) {
      if (numberedCount === 0 && buffer.length > 0 && !isNumberedWorkflowBlock(buffer.join("\n"))) {
        flush();
      }
      numberedCount++;
      buffer.push(line);
      continue;
    }

    if (numberedCount >= 2 && line.trim() && !isHeader) {
      buffer.push(line);
      continue;
    }

    if (line.trim() === "") {
      if (numberedCount >= 2) {
        flush();
      } else if (buffer.length > 0) {
        flush();
      }
      continue;
    }

    if (numberedCount > 0 && numberedCount < 2) {
      numberedCount = 0;
    }

    if (buffer.length > 0 && numberedCount === 0) {
      flush();
    }
    buffer.push(line);
  }

  flush();
  return units.length > 0 ? units : [text.trim()];
}

function isNumberedWorkflowBlock(text: string): boolean {
  const numbered = text.split("\n").filter((line) => /^\s*\d+\.\s+/.test(line));
  return numbered.length >= 2;
}

function hardSplitByTokens(text: string, maxChunkTokens: number): string[] {
  const paragraphs = text.split(/\n{2,}/).map((part) => part.trim()).filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  const flush = () => {
    if (current.trim()) chunks.push(current.trim());
    current = "";
  };

  for (const paragraph of paragraphs) {
    const candidate = current ? `${current}\n\n${paragraph}` : paragraph;
    if (estimateTextTokens(candidate) <= maxChunkTokens) {
      current = candidate;
      continue;
    }
    if (current) flush();
    if (estimateTextTokens(paragraph) <= maxChunkTokens) {
      current = paragraph;
      continue;
    }
    const sentences = paragraph.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [paragraph];
    for (const sentence of sentences) {
      const next = current ? `${current} ${sentence.trim()}` : sentence.trim();
      if (estimateTextTokens(next) <= maxChunkTokens) {
        current = next;
      } else {
        flush();
        current = sentence.trim();
      }
    }
  }
  flush();
  return chunks.length > 0 ? chunks : [text.trim()];
}

function mergeSmallSegments(
  units: Array<{ text: string; reason: SegmentSplitReason }>,
  minChunkTokens: number,
  maxChunkTokens: number
): Array<{ text: string; reason: SegmentSplitReason }> {
  if (units.length <= 1) return units;
  const merged: Array<{ text: string; reason: SegmentSplitReason }> = [];
  let pending: { text: string; reason: SegmentSplitReason } | null = null;

  for (const unit of units) {
    if (!pending) {
      pending = { ...unit };
      continue;
    }
    const combined: string = `${pending.text}\n\n${unit.text}`;
    const pendingSmall = estimateTextTokens(pending.text) < minChunkTokens;
    const unitSmall = estimateTextTokens(unit.text) < minChunkTokens;
    if ((pendingSmall || unitSmall) && estimateTextTokens(combined) <= maxChunkTokens) {
      pending = { text: combined, reason: pending.reason };
      continue;
    }
    merged.push(pending);
    pending = { ...unit };
  }
  if (pending) merged.push(pending);
  return merged;
}
