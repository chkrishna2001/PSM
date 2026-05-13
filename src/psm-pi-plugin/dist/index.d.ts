import { type ModelRuntime } from "psm-sdk";
export interface PsmPluginOptions {
    dbPath: string;
    userId?: string;
    runtime?: ModelRuntime;
}
export declare function createPsmTools(options: PsmPluginOptions): Record<string, (input: Record<string, unknown>) => Promise<unknown>>;
