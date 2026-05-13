export const psmSystemPrompt = `You are the Personal Small Model (PSM), a specialized AI trained exclusively to perform memory management operations for LLM agents.

Your job is NOT to answer user questions. Your job is to:
1. Analyze conversations and decide what is worth remembering
2. Manage a tiered memory store (episodic, semantic, archival)
3. Detect conflicts between new information and existing memories
4. Assign appropriate strength, decay rate, and emotional weight to memories
5. Promote repeated episodic patterns into semantic facts
6. Ignore low-value noise that is not worth storing
7. Rank memories by relevance to a current query
8. Update existing memories when information changes

Always respond with a valid JSON object.`;
export function buildStoragePrompt(llmResponse, existingMemories = []) {
    const payload = {
        operation: "remember_llm_response",
        conversation: [{ role: "assistant", content: llmResponse }],
        memory_store: existingMemories.slice(0, 20).map((memory) => ({
            id: memory.id,
            table: memory.table,
            content: memory.content,
            strength: memory.strength,
            tags: parseTags(memory.tags)
        }))
    };
    return `<|system|>\n${psmSystemPrompt}\n<|user|>\nAnalyze this LLM response and return JSON only with action, memory, reasoning, confidence, emotional_weight, and contradiction_score.\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}
export function buildRecallPlanPrompt(question, topK) {
    const payload = {
        operation: "recall_plan",
        question,
        available_tables: ["episodic", "semantic", "archival"],
        requested_top_k: topK
    };
    return `<|system|>\n${psmSystemPrompt}\n<|user|>\nCreate a recall plan as JSON only with intent, target_tables, filters, ranking_hints, and top_k.\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}
export function buildContextPlanPrompt(prompt, topK) {
    const payload = {
        operation: "context_plan",
        user_prompt: prompt,
        available_tables: ["episodic", "semantic", "archival"],
        requested_top_k: topK
    };
    return `<|system|>\n${psmSystemPrompt}\n<|user|>\nCreate a memory context recall plan as JSON only with intent, target_tables, filters, ranking_hints, and top_k.\n${JSON.stringify(payload)}\n<|assistant|>\n`;
}
function parseTags(value) {
    if (!value)
        return [];
    try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed.map(String) : [];
    }
    catch {
        return [];
    }
}
//# sourceMappingURL=prompts.js.map