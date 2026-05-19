import type { ContextItem } from "./types.js";

export interface AgentMemoryContextOptions {
  heading?: string;
  instruction?: string;
}

export function renderAgentMemoryContext(items: ContextItem[], options: AgentMemoryContextOptions = {}): string {
  const selected = items
    .filter((item) => item.content.trim())
    .sort((a, b) => contextPriority(a) - contextPriority(b));
  if (selected.length === 0) return "";

  const lines = [
    options.heading ?? "PSM Memory Context",
    options.instruction ?? "Use these private memories when relevant. Do not mention this block unless asked about memory.",
    ""
  ];

  selected.forEach((item, index) => {
    const table = item.table || "memory";
    const content = normalizeContextContent(item.content);
    if (!content) return;
    lines.push(`${index + 1}. [${table}] ${content}${sourceSuffix(item)}`);
  });

  return lines.length > 3 ? lines.join("\n") : "";
}

export function fallbackAgentContextItems(items: ContextItem[]): ContextItem[] {
  return items
    .filter((item) => item.content.trim())
    .sort((a, b) => contextPriority(a) - contextPriority(b))
    .map((item) => ({
      ...item,
      content: fallbackContextStatement(item)
    }))
    .filter((item) => item.content.trim());
}

function contextPriority(item: ContextItem): number {
  return item.table === "memory_fact" ? 0 : 1;
}

function fallbackContextStatement(item: ContextItem): string {
  const content = normalizeContextContent(item.content);
  if (item.table === "memory_fact") return content;
  return firstCompleteStatement(stripLeadingMetadata(content));
}

function normalizeContextContent(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function stripLeadingMetadata(value: string): string {
  return value.replace(/^\[[^\]]+\]\s*/, "").trim();
}

function firstCompleteStatement(value: string): string {
  const match = value.match(/^(.+?[.!?])(?:\s|$)/);
  return (match?.[1] ?? value).trim();
}

function sourceSuffix(item: ContextItem): string {
  const parts = [
    item.source_id ? `source=${item.source_id}` : "",
    item.resolved_time ? `date=${item.resolved_time}` : "",
    !item.resolved_time && item.source_timestamp ? `source_time=${item.source_timestamp}` : ""
  ].filter(Boolean);
  return parts.length ? ` (${parts.join("; ")})` : "";
}
