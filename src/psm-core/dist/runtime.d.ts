import type { ModelRuntime } from "./types.js";
export declare class HeuristicRuntime implements ModelRuntime {
    generateJson(prompt: string): Promise<string>;
}
