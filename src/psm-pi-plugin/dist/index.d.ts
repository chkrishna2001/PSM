import { type ModelRuntime } from "@psm-memory/sdk";
export interface PsmPluginOptions {
    dbPath: string;
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
export declare function createPsmTools(options: PsmPluginOptions): Record<string, (input: Record<string, unknown>) => Promise<unknown>>;
export declare function createPsmHooks(options: PsmPluginOptions): PsmHooks;
