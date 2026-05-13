import type { MemoryAction, MemoryRoute } from "./types.js";
export declare function normalizeAction(action: unknown): MemoryAction;
export declare function routeForAction(action: MemoryAction): MemoryRoute;
export declare function actionFromOperation(operation: string): MemoryAction;
