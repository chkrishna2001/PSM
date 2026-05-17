import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, NodeLlamaRuntime, PsmService, readPsmConfig, resolvePsmDbPath, type ModelRuntime } from "@psm-memory/sdk";

export interface PsmPluginOptions {
  dbPath?: string;
  userId?: string;
  runtime?: ModelRuntime;
  modelPath?: string;
  topK?: number;
  onMemoryWriteError?: (error: unknown) => void;
}

export interface PsmBeforePromptInput {
  prompt: string;
  userId?: string;
  topK?: number;
}

export interface PsmBeforePromptResult {
  userId: string;
  prompt: string;
  contextMessage: PsmChatMessage | null;
  messages: PsmChatMessage[];
  memoryContext: string;
  rawContext: Record<string, unknown>;
}

export interface PsmAfterResponseInput {
  response?: string;
  decision?: unknown;
  userId?: string;
}

export interface PsmChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface PsmHooks {
  enrichPrompt(input: PsmBeforePromptInput): Promise<PsmBeforePromptResult>;
  rememberResponse(input: PsmAfterResponseInput): void;
  beforePrompt(input: PsmBeforePromptInput): Promise<PsmBeforePromptResult>;
  afterResponse(input: PsmAfterResponseInput): void;
  flush(): Promise<void>;
  close(): Promise<void>;
}

export function createPsmTools(options: PsmPluginOptions): Record<string, (input: Record<string, unknown>) => Promise<unknown>> {
  const { service, defaultUser } = createService(options);

  return {
    "psm.remember": async (input) => service.remember({
      llmResponse: requireString(input, "llm_response"),
      userId: stringOr(input.user, defaultUser)
    }),
    "psm.recall": async (input) => service.recall({
      question: requireString(input, "question"),
      userId: stringOr(input.user, defaultUser),
      topK: numberOr(input.top_k, options.topK ?? 5)
    })
  };
}

export function createPsmHooks(options: PsmPluginOptions): PsmHooks {
  const { store, service, defaultUser } = createService(options);
  const pending = new Set<Promise<void>>();

  const enrichPrompt = async (input: PsmBeforePromptInput): Promise<PsmBeforePromptResult> => {
    const prompt = requireValue(input.prompt, "prompt");
    const userId = stringOr(input.userId, defaultUser);
    const rawContext = await service.context({
      prompt,
      userId,
      topK: input.topK ?? options.topK ?? 5
    });
    const memoryContext = renderMemoryContext(rawContext);
    const contextMessage = memoryContext ? {
      role: "system" as const,
      content: memoryContext
    } : null;

    return {
      userId,
      prompt,
      contextMessage,
      messages: contextMessage ? [contextMessage, { role: "user", content: prompt }] : [{ role: "user", content: prompt }],
      memoryContext,
      rawContext
    };
  };

  const rememberResponse = (input: PsmAfterResponseInput): void => {
    const llmResponse = renderResponseForStorage(input);
    if (!llmResponse) return;

    const task = service.remember({
      llmResponse,
      userId: stringOr(input.userId, defaultUser),
      source: {
        source_kind: "pi-plugin",
        source_timestamp: new Date().toISOString(),
        source_label: "PI plugin response"
      }
    }).then(() => undefined);

    pending.add(task);
    task.catch((error) => {
      options.onMemoryWriteError?.(error);
    }).finally(() => {
      pending.delete(task);
    });
  };

  return {
    enrichPrompt,
    rememberResponse,
    beforePrompt: enrichPrompt,
    afterResponse: rememberResponse,

    async flush() {
      await Promise.allSettled([...pending]);
    },

    async close() {
      await Promise.allSettled([...pending]);
      store.close();
    }
  };
}

function createService(options: PsmPluginOptions): { store: MemoryStore; service: PsmService; defaultUser: string } {
  const dbPath = resolvePsmDbPath({ dbPath: options.dbPath });
  mkdirSync(dirname(dbPath), { recursive: true });
  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const service = new PsmService(store, resolveRuntime(options));
  const defaultUser = options.userId ?? readPsmConfig().userId;

  return { store, service, defaultUser };
}

function resolveRuntime(options: PsmPluginOptions): ModelRuntime {
  if (options.runtime) return options.runtime;
  if (options.modelPath) return new NodeLlamaRuntime({ modelPath: options.modelPath });
  throw new Error("PSM model runtime is required. Pass runtime or modelPath to createPsmHooks.");
}

function renderMemoryContext(rawContext: Record<string, unknown>): string {
  const memories = Array.isArray(rawContext.context_items) ? rawContext.context_items : [];
  if (memories.length === 0) return "";

  const lines = memories.map((memory, index) => {
    const item = memory as Record<string, unknown>;
    const content = typeof item.content === "string" ? item.content : "";
    return `${index + 1}. ${content}`;
  }).filter((line) => line.trim());

  if (lines.length === 0) return "";
  return [
    "PSM Memory Context",
    "Use these retrieved memories as private context. Do not mention this block unless the user asks about memory.",
    "",
    ...lines
  ].join("\n");
}

function renderResponseForStorage(input: PsmAfterResponseInput): string {
  const parts: string[] = [];
  if (typeof input.response === "string" && input.response.trim()) {
    parts.push(`LLM response:\n${input.response.trim()}`);
  }
  if (input.decision !== undefined && input.decision !== null) {
    parts.push(`LLM decision:\n${stringifyDecision(input.decision)}`);
  }
  return parts.join("\n\n");
}

function stringifyDecision(value: unknown): string {
  if (typeof value === "string") return value.trim();
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function requireValue(value: unknown, key: string): string {
  if (typeof value === "string" && value.trim()) return value;
  throw new Error(`Missing required hook input: ${key}`);
}

function requireString(input: Record<string, unknown>, key: string): string {
  const value = input[key];
  if (typeof value === "string" && value.trim()) return value;
  throw new Error(`Missing required tool input: ${key}`);
}

function stringOr(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function numberOr(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
