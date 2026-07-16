using System.ComponentModel;
using ModelContextProtocol.Server;
using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;

namespace PsmMemory.Mcp;

/// <summary>
/// Thin MCP tool wrappers around the shared <see cref="PsmService"/> singleton (constructed once
/// at process startup in Program.cs and injected here via DI). Each tool method just maps its MCP
/// arguments onto the corresponding Core request DTO, awaits the call, and returns the result --
/// no business logic lives here.
/// </summary>
[McpServerToolType]
public sealed class PsmMemoryTools
{
    private const string DomainDescription =
        "Which trained-adapter domain to use: \"coding\" (default) or \"conversational\". " +
        "\"conversational\" requires conversational_*.onnx_adapter files to exist in the model " +
        "directory (not yet trained as of this writing) -- requesting it before those exist fails " +
        "with a clear error.";

    private readonly PsmService _psm;

    public PsmMemoryTools(PsmService psm)
    {
        _psm = psm;
    }

    private static PsmDomain ParseDomain(string domain) => domain.Trim().ToLowerInvariant() switch
    {
        "coding" => PsmDomain.Coding,
        "conversational" => PsmDomain.Conversational,
        _ => throw new ArgumentException($"domain must be one of coding|conversational, got '{domain}'.", nameof(domain)),
    };

    [McpServerTool(Name = "remember")]
    [Description(
        "Extract and store durable memory from an assistant/LLM response. The PSM storage model " +
        "decides whether to ignore it, store it as an episodic memory, or promote it to semantic " +
        "memory, and may merge it with an existing related memory. Returns the storage decision " +
        "(action, route, what was written, and the resulting memory payload).")]
    public async Task<RememberResult> Remember(
        [Description("The assistant/LLM response text to extract durable memory from -- this is the content that gets remembered.")]
        string llmResponse,
        [Description("The user id memories are scoped to.")]
        string userId,
        [Description("The user message that prompted the response, kept for conversational context (not currently used to influence the storage decision itself).")]
        string? userMessage = null,
        [Description(DomainDescription)]
        string domain = "coding",
        CancellationToken ct = default)
    {
        var result = await _psm.RememberAsync(new RememberRequest
        {
            LlmResponse = llmResponse,
            UserMessage = userMessage,
            UserId = userId,
            Domain = ParseDomain(domain)
        }, ct).ConfigureAwait(false);

        return result;
    }

    [McpServerTool(Name = "recall")]
    [Description(
        "Answer a specific question by planning a memory retrieval (which memory tables/tiers to " +
        "search) and returning the most relevant stored memories for that user, ranked by hybrid " +
        "relevance score.")]
    public async Task<RecallResult> Recall(
        [Description("The question to retrieve relevant memories for.")]
        string question,
        [Description("The user id memories are scoped to.")]
        string userId,
        [Description("Maximum number of memories to return. Defaults to 5 if omitted.")]
        int? topK = null,
        [Description(DomainDescription)]
        string domain = "coding",
        CancellationToken ct = default)
    {
        var result = await _psm.RecallAsync(new RecallRequest
        {
            Question = question,
            UserId = userId,
            TopK = topK,
            Domain = ParseDomain(domain)
        }, ct).ConfigureAwait(false);

        return result;
    }

    [McpServerTool(Name = "context")]
    [Description(
        "Given an upcoming prompt/conversation turn, retrieve the stored memories most relevant as " +
        "grounding context for it (same recall-plan-then-rank flow as `recall`, tuned with a lower " +
        "relevance threshold appropriate for general context injection rather than a targeted " +
        "question).")]
    public async Task<RecallResult> Context(
        [Description("The upcoming prompt/conversation turn to retrieve grounding context for.")]
        string prompt,
        [Description("The user id memories are scoped to.")]
        string userId,
        [Description("Maximum number of memories to return. Defaults to 5 if omitted.")]
        int? topK = null,
        [Description(DomainDescription)]
        string domain = "coding",
        CancellationToken ct = default)
    {
        var result = await _psm.ContextAsync(new ContextRequest
        {
            Prompt = prompt,
            UserId = userId,
            TopK = topK,
            Domain = ParseDomain(domain)
        }, ct).ConfigureAwait(false);

        return result;
    }
}
