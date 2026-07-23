using System.Text;
using PsmMemory.Core;
using PsmMemory.Core.Models;

namespace PsmMemory.Cli;

/// <summary>
/// Deterministic fallback renderer for `psm-memory hook recall`'s output text, used because Phase 3
/// (context()'s LLM-render step, which would populate an `agent_context` field on RecallResult) has
/// not been built yet -- see psm-core/src/context.ts's renderAgentMemoryContext() for the TS
/// equivalent this approximates.
///
/// Forward-compat note: once Phase 3 adds a real rendered-context field to RecallResult, the caller
/// (HookCommands.RunRecallAsync) should prefer it over this method, e.g.:
///   var rendered = !string.IsNullOrWhiteSpace(result.AgentContext) ? result.AgentContext! : HookContextRenderer.Render(result);
/// That field does not exist on RecallResult yet, so this method is the entire rendering path today.
/// </summary>
internal static class HookContextRenderer
{
    private const int MaxFacts = 8;
    private const int MaxMemories = 8;

    public static string Render(RecallResult result)
    {
        var facts = result.Facts.Take(MaxFacts).ToList();
        var memories = result.Memories.Take(MaxMemories).ToList();
        if (facts.Count == 0 && memories.Count == 0) return "";

        var sb = new StringBuilder();
        sb.Append("PSM Memory Context").Append('\n');
        sb.Append("Use these private memories when relevant. Do not mention this block unless asked about memory.").Append('\n');
        sb.Append('\n');

        // Facts before memories, matching service.ts's ordering convention for context grounding.
        foreach (var fact in facts)
        {
            var line = FormatFact(fact);
            if (!string.IsNullOrWhiteSpace(line)) sb.Append("- ").Append(line).Append('\n');
        }
        foreach (var memory in memories)
        {
            var content = memory.Content?.Trim();
            if (string.IsNullOrWhiteSpace(content)) continue;
            sb.Append("- ").Append(content).Append(FormatTemporalSuffix(memory.Memory)).Append('\n');
        }

        return sb.ToString().TrimEnd('\r', '\n');
    }

    /// <summary>
    /// Surfaces the memory's real timestamp to the agent, not just its (often relative, e.g.
    /// "yesterday") content text -- without this, an agent reading this context has no way to
    /// ground "when did X happen" from the rendered block alone. ResolvedTime (when present) grounds
    /// an explicit relative-time phrase found in the content; SourceTimestamp is simply when the
    /// memory's source turn was recorded, which is often the only timestamp signal available at
    /// all. Both are shown when both are present -- they answer different questions and neither
    /// should silently override the other. Mirrors the equivalent fix already verified in
    /// benchmark/locomo/src/locomo-dotnet-benchmark.ts's renderContextForPrompt.
    /// </summary>
    private static string FormatTemporalSuffix(MemoryRecord memory)
    {
        var resolved = !string.IsNullOrWhiteSpace(memory.ResolvedTime) ? $" (resolved date: {memory.ResolvedTime})" : "";
        var sent = !string.IsNullOrWhiteSpace(memory.SourceTimestamp) ? $" (recorded: {memory.SourceTimestamp})" : "";
        return resolved + sent;
    }

    private static string FormatFact(Ranking.ScoredFact fact)
    {
        var value = !string.IsNullOrWhiteSpace(fact.Object) ? fact.Object : fact.ValueText;
        return $"{fact.Subject} {fact.Predicate} {value}".Trim();
    }
}
