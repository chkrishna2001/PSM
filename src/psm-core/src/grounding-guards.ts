import { routeForAction } from "./actions.js";
import { tokenize } from "./ranking.js";
import type { StorageDecision } from "./types.js";

const BLEED_PATTERN = /checkpoint|powershell|gate datasets|nvidia-smi|direct probe|token budget|runpod|fact parser|malformed parser|constoursated|gate6|expanded probe|gate-?\d/i;

export type StorageGuardRejectRoute = "grounding_reject" | "grounding_reject_bleed";

export interface StorageGuardResult {
  decision: StorageDecision;
  rejected: boolean;
  guard_route?: StorageGuardRejectRoute;
  guard_reason?: string;
}

export function hasCurriculumBleed(text: string): boolean {
  return BLEED_PATTERN.test(text);
}

export function significantTokens(text: string): string[] {
  return tokenize(text).filter((token) => token.length >= 3 && !/^\d+$/.test(token));
}

export function groundingOverlapScore(rememberTarget: string, storedText: string): { overlap: number; required: number; grounded: boolean } {
  const inputTokens = significantTokens(rememberTarget);
  if (inputTokens.length === 0) {
    return { overlap: 0, required: 0, grounded: true };
  }
  const storedSet = new Set(significantTokens(storedText));
  const overlap = inputTokens.filter((token) => storedSet.has(token)).length;
  const required = Math.min(2, Math.max(1, Math.ceil(inputTokens.length * 0.1)));
  return { overlap, required, grounded: overlap >= required };
}

export function isGroundedInSource(rememberTarget: string, storedText: string): boolean {
  return groundingOverlapScore(rememberTarget, storedText).grounded;
}

function storedTextFromDecision(decision: StorageDecision): string {
  const content = decision.memory?.content?.trim() ?? "";
  const factParts = (decision.facts ?? []).flatMap((fact) => [
    fact.subject,
    fact.predicate,
    fact.value_text,
    fact.evidence_text ?? ""
  ]);
  return [content, ...factParts].filter(Boolean).join(" ");
}

function wouldPersist(decision: StorageDecision): boolean {
  if (decision.parse_error) return false;
  const route = routeForAction(decision.action);
  if (route === "ignore" || route === "recall_only") return false;
  return Boolean(storedTextFromDecision(decision).trim());
}

export function applyStorageGuards(rememberTarget: string, decision: StorageDecision): StorageGuardResult {
  if (!wouldPersist(decision)) {
    return { decision, rejected: false };
  }
  const storedText = storedTextFromDecision(decision);
  if (hasCurriculumBleed(storedText)) {
    return {
      decision,
      rejected: true,
      guard_route: "grounding_reject_bleed",
      guard_reason: "Stored content matches curriculum bleed blocklist."
    };
  }
  if (!isGroundedInSource(rememberTarget, storedText)) {
    return {
      decision,
      rejected: true,
      guard_route: "grounding_reject",
      guard_reason: "Stored content is not grounded in remember_target tokens."
    };
  }
  return { decision, rejected: false };
}

export function isFailSafeIgnore(result: Record<string, unknown>): boolean {
  const route = typeof result.route === "string" ? result.route : "";
  const reasoning = typeof result.reasoning === "string" ? result.reasoning : "";
  if (route === "parse_error_noop") return true;
  return /model output unparseable|storing nothing/i.test(reasoning);
}
