import { type ContextRender, type RecallPlan, type StorageDecision } from "./types.js";
export declare function extractJsonObject(text: string): string | null;
export declare function parseStorageDecision(rawText: string, fallbackContent: string, fallbackAction?: string): StorageDecision;
export declare function parseRecallPlan(rawText: string, question: string, topK?: number): RecallPlan;
export declare function parseContextRender(rawText: string, topK?: number): ContextRender;
