const stopwords = new Set([
    "the", "and", "for", "that", "this", "with", "you", "your", "what", "when", "where", "why", "how", "are",
    "was", "were", "has", "have", "had", "from", "about", "into", "onto", "then", "than", "they", "them"
]);
export function rankMemories(query, memories, topK) {
    const qTokens = tokenize(query);
    const ranked = memories.map((memory) => {
        const score = lexicalScore(qTokens, tokenize(memory.content)) +
            0.15 * (memory.confidence ?? 0.5) +
            0.1 * (memory.strength ?? 0.5) +
            0.05 * (memory.table === "semantic" ? 1 : 0);
        return {
            ...memory,
            score: Number(score.toFixed(6)),
            metadata: {
                tags: parseJson(memory.tags),
                source_episodes: parseJson(memory.source_episodes)
            }
        };
    });
    return ranked.sort((a, b) => b.score - a.score).slice(0, topK);
}
export function tokenize(text) {
    return text
        .toLowerCase()
        .match(/[a-z0-9]+/g)
        ?.map(normalizeToken)
        .filter((token) => token.length > 2 && !stopwords.has(token)) ?? [];
}
function lexicalScore(queryTokens, memoryTokens) {
    if (queryTokens.length === 0 || memoryTokens.length === 0)
        return 0;
    const memorySet = new Set(memoryTokens);
    const overlap = queryTokens.filter((token) => memorySet.has(token)).length;
    return overlap / Math.sqrt(queryTokens.length * memoryTokens.length);
}
function parseJson(value) {
    if (!value)
        return [];
    try {
        return JSON.parse(value);
    }
    catch {
        return [];
    }
}
function normalizeToken(token) {
    if (token.endsWith("ies") && token.length > 4)
        return `${token.slice(0, -3)}y`;
    if (token.endsWith("es") && token.length > 4)
        return token.slice(0, -2);
    if (token.endsWith("s") && token.length > 3)
        return token.slice(0, -1);
    return token;
}
//# sourceMappingURL=ranking.js.map