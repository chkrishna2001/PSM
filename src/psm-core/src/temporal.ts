import type { MemoryFactPayload, MemoryPayload } from "./types.js";

const relativePattern = /\b(yesterday|today|tomorrow|last week|next week|last month|next month|last year|next year)\b/i;
const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

export function normalizeMemoryTemporalFields(memory: MemoryPayload, sourceTimestamp?: string): MemoryPayload {
  const temporalExpression = memory.temporal_expression ?? detectRelativeExpression(memory.content);
  const resolved = temporalExpression && sourceTimestamp ? resolveRelativeTime(temporalExpression, sourceTimestamp) : undefined;
  if (!temporalExpression) return memory;
  if (!isSupportedTemporalExpression(temporalExpression)) {
    return {
      ...memory,
      temporal_expression: undefined,
      resolved_time: undefined,
      resolved_time_confidence: undefined
    };
  }
  if (!resolved) return memory;
  return {
    ...memory,
    temporal_expression: memory.temporal_expression ?? temporalExpression,
    resolved_time: resolved,
    resolved_time_confidence: Math.max(memory.resolved_time_confidence ?? 0, 0.9)
  };
}

export function normalizeFactTemporalFields(fact: MemoryFactPayload, sourceTimestamp?: string): MemoryFactPayload {
  const text = [fact.evidence_text, fact.value_text, typeof fact.value === "string" ? fact.value : ""].filter(Boolean).join(" ");
  const temporalExpression = fact.temporal_expression ?? detectRelativeExpression(text);
  const resolved = temporalExpression && sourceTimestamp ? resolveRelativeTime(temporalExpression, sourceTimestamp) : undefined;
  if (!temporalExpression) return fact;
  if (!isSupportedTemporalExpression(temporalExpression)) {
    return {
      ...fact,
      temporal_expression: undefined,
      resolved_time: undefined,
      resolved_time_confidence: undefined
    };
  }
  if (!resolved) return fact;
  return {
    ...fact,
    temporal_expression: fact.temporal_expression ?? temporalExpression,
    resolved_time: resolved,
    resolved_time_confidence: Math.max(fact.resolved_time_confidence ?? 0, 0.9)
  };
}

function isSupportedTemporalExpression(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return relativePattern.test(normalized)
    || /\b\d{4}\b/.test(normalized)
    || /\b\d{1,2}\s+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b/i.test(normalized);
}

export function detectRelativeExpression(text: string | undefined): string | undefined {
  return text?.match(relativePattern)?.[1]?.toLowerCase();
}

export function resolveRelativeTime(expression: string, sourceTimestamp: string): string | undefined {
  const anchor = parseSourceDate(sourceTimestamp);
  if (!anchor) return undefined;
  const normalized = expression.toLowerCase();
  if (normalized === "today") return formatDate(anchor);
  if (normalized === "yesterday") return formatDate(addDays(anchor, -1));
  if (normalized === "tomorrow") return formatDate(addDays(anchor, 1));
  if (normalized === "last week") return `week before ${formatDate(anchor)}`;
  if (normalized === "next week") return `week after ${formatDate(anchor)}`;
  if (normalized === "last month") return formatMonth(addMonths(anchor, -1));
  if (normalized === "next month") return formatMonth(addMonths(anchor, 1));
  if (normalized === "last year") return String(anchor.getUTCFullYear() - 1);
  if (normalized === "next year") return String(anchor.getUTCFullYear() + 1);
  return undefined;
}

function parseSourceDate(value: string): Date | undefined {
  const direct = new Date(value);
  if (!Number.isNaN(direct.getTime())) return utcDate(direct.getUTCFullYear(), direct.getUTCMonth(), direct.getUTCDate());
  const match = value.match(/\b(\d{1,2})\s+(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December),?\s+(\d{4})\b/i);
  if (!match) return undefined;
  const day = Number(match[1]);
  const month = monthIndex(match[2]);
  const year = Number(match[3]);
  if (!Number.isInteger(day) || month < 0 || !Number.isInteger(year)) return undefined;
  return utcDate(year, month, day);
}

function monthIndex(value: string): number {
  const normalized = value.toLowerCase();
  return monthNames.findIndex((month) => month.toLowerCase().startsWith(normalized.slice(0, 3)));
}

function addDays(date: Date, days: number): Date {
  return utcDate(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate() + days);
}

function addMonths(date: Date, months: number): Date {
  return utcDate(date.getUTCFullYear(), date.getUTCMonth() + months, 1);
}

function utcDate(year: number, month: number, day: number): Date {
  return new Date(Date.UTC(year, month, day));
}

function formatDate(date: Date): string {
  return `${date.getUTCDate()} ${monthNames[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

function formatMonth(date: Date): string {
  return `${monthNames[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}
