import { buildContextPlanPrompt, buildRecallPlanPrompt, buildStoragePrompt } from "./prompts.js";
import { parseRecallPlan, parseStorageDecision } from "./json.js";
import { rankMemories } from "./ranking.js";
export class PsmService {
    store;
    runtime;
    constructor(store, runtime) {
        this.store = store;
        this.runtime = runtime;
    }
    async context(request) {
        const topK = request.topK ?? 5;
        const raw = await this.runtime.generateJson(buildContextPlanPrompt(request.prompt, topK), { temperature: 0, maxTokens: 256 });
        const plan = parseRecallPlan(raw, request.prompt, topK);
        const memories = this.store.selectMemories(request.userId, plan.target_tables, Math.max(100, plan.top_k * 10));
        const ranked = rankMemories([...plan.ranking_hints, request.prompt].join(" "), memories, plan.top_k);
        this.store.updateAccess(ranked);
        return {
            user_id: request.userId,
            prompt: request.prompt,
            recall_plan: plan,
            memory_context: ranked.map((memory) => ({
                table: memory.table,
                id: memory.id,
                content: memory.content,
                score: memory.score,
                metadata: memory.metadata
            }))
        };
    }
    async recall(request) {
        const topK = request.topK ?? 5;
        const raw = await this.runtime.generateJson(buildRecallPlanPrompt(request.question, topK), { temperature: 0, maxTokens: 256 });
        const plan = parseRecallPlan(raw, request.question, topK);
        const memories = this.store.selectMemories(request.userId, plan.target_tables, Math.max(100, plan.top_k * 10));
        const ranked = rankMemories([...plan.ranking_hints, request.question].join(" "), memories, plan.top_k);
        this.store.updateAccess(ranked);
        return {
            user_id: request.userId,
            question: request.question,
            recall_plan: plan,
            memories: ranked.map((memory) => ({
                table: memory.table,
                id: memory.id,
                content: memory.content,
                score: memory.score,
                metadata: memory.metadata
            }))
        };
    }
    async remember(request) {
        const existing = this.store.selectMemories(request.userId, ["semantic", "episodic"], 50);
        const raw = await this.runtime.generateJson(buildStoragePrompt(request.llmResponse, existing), { temperature: 0, maxTokens: 256 });
        const decision = parseStorageDecision(raw, request.llmResponse, "store_episodic");
        const result = this.store.applyDecision(request.userId, "llm-response", decision);
        return {
            user_id: request.userId,
            action: result.action,
            route: result.route,
            written: result.written,
            memory: decision.memory,
            reasoning: decision.reasoning,
            raw_model_json: decision.raw_json,
            parse_error: decision.parse_error
        };
    }
}
//# sourceMappingURL=service.js.map