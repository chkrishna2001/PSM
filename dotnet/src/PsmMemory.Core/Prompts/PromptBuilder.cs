using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using PsmMemory.Core.Models;

namespace PsmMemory.Core.Prompts;

/// <summary>
/// Builds ChatML prompts for the three PSM ONNX adapters (storage, retrieval_plan, consolidation).
///
/// IMPORTANT DEVIATION FROM psm-core/src/prompts.ts: this does NOT reproduce prompts.ts's
/// buildStoragePrompt/buildRecallPlanPrompt/buildContextPlanPrompt byte-for-byte. Those functions
/// targeted the OLD Python-subprocess PsmModelRuntime lineage (`&lt;|system|&gt;`/`&lt;|user|&gt;`/`&lt;|assistant|&gt;`
/// bare markers, a rich JSON payload including memory_store/source metadata for the storage prompt).
///
/// The ONNX adapters actually loaded at psm-model/prod-memory/onnx-runtime/v1 (validated this
/// session at 0.84 gate-score parity) were trained on a DIFFERENT, simpler format defined by
/// psm-model/src/psm_model/prompts.py's system instructions plus
/// psm-model/prod-memory/prod_memory/hf_prompts.py's row_messages() user-text templates, and
/// exported with a Qwen2 ChatML chat template (`&lt;|im_start|&gt;role\n...&lt;|im_end|&gt;\n`) — confirmed
/// by cross-referencing psm-model/prod-memory/onnx-runtime/v1/chat_template.jinja and the exact
/// prompts baked into psm-model/prod-memory/onnx-spike/gguf-spike/gate-cases-prompts.json (the file
/// this session's FoundrySpike gate harness used to hit 0.84). This class reproduces THAT format,
/// because that is the format the on-disk adapters actually understand — using prompts.ts's format
/// here would compile fine but silently produce garbage completions.
/// </summary>
public static class PromptBuilder
{
    // --- Storage adapter (psm_model.prompts.JSON_SYSTEM_INSTRUCTION) ---------------------------

    public const string StorageSystemInstruction =
        "You are the PSM storage model.\n" +
        "Return one strict JSON object compatible with the PSM StorageDecision schema.\n" +
        "First write reasoning explaining your decision, then action, memory (or null),\n" +
        "facts[], and indexables[] in that order.\n" +
        "Do not include markdown, prose, comments, or fallback text outside JSON.\n" +
        "Facts must be explicit and supported by evidence_text from the current input.";

    private const string StorageUserInstruction =
        "Extract durable memory from the assistant response below.\n" +
        "Choose ignore, store_episodic, or promote_semantic.\n" +
        "When storing, emit grounded memory.content, facts[], and indexables[] from the text.\n\n";

    /// <summary>
    /// 2026-07-23: label for the optional context block in <see cref="BuildStoragePrompt"/>. Wording
    /// is deliberate, not decorative -- a naive "just prepend prior turns" experiment (LoCoMo
    /// benchmark harness, same date) was verified live against the real model to cause the adapter to
    /// extract facts from ANYWHERE in the window and misattribute them to the current turn's source
    /// id. This label exists specifically to train the discipline that experiment lacked: use context
    /// to understand the current turn, never extract FROM the context itself.
    /// </summary>
    private const string StorageContextLabel =
        "Recent context, oldest first (for understanding only -- do NOT extract a memory from this " +
        "section; base your decision only on the assistant response below):\n";

    private const string StorageResponseLabel = "Assistant response:\n";

    /// <summary>
    /// Builds the storage-decision prompt. Ported from prod_memory.hf_prompts.storage_inference_messages
    /// (system=JSON_SYSTEM_INSTRUCTION, user=StorageUserPrefix+llmResponse), ChatML-wrapped.
    /// Note: unlike psm-core's buildStoragePrompt, the real trained format does NOT embed existing
    /// memories, source metadata, or userMessage in the storage prompt at all — the model only ever
    /// sees the assistant response text. Existing-memory awareness happens later, via the separate
    /// consolidation adapter (see BuildConsolidationPrompt) — that's a deliberate two-step design,
    /// not an omission.
    ///
    /// <paramref name="contextTurns"/> is new (2026-07-23, not yet trained on): when null/empty, this
    /// produces a BYTE-IDENTICAL prompt to the original single-argument form -- the currently-running
    /// production adapters (which have never seen a context block) are completely unaffected. Only
    /// once a context-aware adapter is trained on this exact format should callers start passing a
    /// non-empty context.
    /// </summary>
    public static string BuildStoragePrompt(string llmResponse, IReadOnlyList<string>? contextTurns = null)
    {
        var user = StorageUserInstruction;
        if (contextTurns is { Count: > 0 })
        {
            user += StorageContextLabel + string.Join("\n", contextTurns) + "\n\n";
        }
        user += StorageResponseLabel + llmResponse.Trim();
        return ChatMl(StorageSystemInstruction, user);
    }

    /// <summary>
    /// Repair-retry prompt for when the storage adapter's first output fails to parse. No dedicated
    /// repair adapter exists in the 3-adapter ONNX runtime (storage/retrieval_plan/consolidation only),
    /// so this reuses the storage adapter itself with an augmented instruction — adapted from the
    /// spirit of psm-core's buildStorageRepairPrompt, but written as a plain-language ChatML user
    /// message (not the old JSON "repair_remember_json" operation payload, which the ONNX storage
    /// adapter was never trained to recognize).
    /// </summary>
    public static string BuildStorageRepairPrompt(string llmResponse, string invalidOutput)
    {
        var user =
            StorageUserInstruction + StorageResponseLabel + llmResponse.Trim() + "\n\n" +
            "Your previous answer was not valid JSON and could not be parsed:\n" +
            invalidOutput + "\n\n" +
            "Re-answer now. Return exactly one valid JSON object and nothing else.";
        return ChatMl(StorageSystemInstruction, user);
    }

    // --- Retrieval-plan adapter (psm_model.prompts.RECALL_SYSTEM_INSTRUCTION) ------------------

    public const string RecallSystemInstruction =
        "You are the PSM memory planner.\n" +
        "Return one strict JSON recall plan object with intent, target_tables, filters, ranking_hints, temporal_intent, and top_k.\n" +
        "Choose memory tiers from episodic, semantic, and archival. Do not answer the user.";

    private const string RecallUserInstruction =
        "Create a recall plan as JSON only with intent, target_tables, filters, ranking_hints, " +
        "temporal_intent, and top_k. PSM owns memory planning; do not answer the user.\n";

    private static readonly string[] AvailableTables = { "episodic", "semantic", "archival" };

    /// <summary>
    /// Builds the recall-plan prompt for RecallAsync (question-driven). Ported from
    /// prod_memory.hf_prompts.row_messages's "recall_plan" branch.
    /// </summary>
    public static string BuildRecallPlanPrompt(string question, int topK)
    {
        var payload = SortedJson(new Dictionary<string, object?>
        {
            ["operation"] = "recall_plan",
            ["available_tables"] = AvailableTables,
            ["requested_top_k"] = topK,
            ["question"] = question
        });
        return ChatMl(RecallSystemInstruction, RecallUserInstruction + payload);
    }

    /// <summary>
    /// Builds the recall-plan prompt for ContextAsync (prompt-driven). Ported from
    /// prod_memory.hf_prompts.row_messages's "context_plan" branch — same user instruction text as
    /// recall_plan (the real training data unifies both under one wording; only the payload's
    /// "user_prompt" vs "question" field differs, matching prod_memory/build_recall_locomo_rows.py's
    /// _recall_row()).
    /// </summary>
    public static string BuildContextPlanPrompt(string prompt, int topK)
    {
        var payload = SortedJson(new Dictionary<string, object?>
        {
            ["operation"] = "context_plan",
            ["available_tables"] = AvailableTables,
            ["requested_top_k"] = topK,
            ["user_prompt"] = prompt
        });
        return ChatMl(RecallSystemInstruction, RecallUserInstruction + payload);
    }

    // --- Consolidation adapter (psm_model.prompts.CONSOLIDATION_SYSTEM_INSTRUCTION) -------------

    public const string ConsolidationSystemInstruction =
        "You are the PSM memory consolidator.\n" +
        "Given a new memory and one existing memory a retrieval step surfaced as related, decide whether " +
        "to store_episodic (the new memory is independent), update_existing (it restates or elaborates the " +
        "same fact -- merge into one updated memory), or flag_conflict (it contradicts the existing memory).\n" +
        "Return one strict JSON object. First write reasoning explaining your decision, then action,\n" +
        "target_memory_id, and merged_content in that order.";

    private const string ConsolidationUserInstruction =
        "Decide store_episodic, update_existing, or flag_conflict as JSON only. " +
        "First write reasoning, then action, target_memory_id, and merged_content.\n";

    /// <summary>
    /// Builds the consolidation-decision prompt. Ported from prod_memory.build_consolidation_rows._row
    /// + prod_memory.hf_prompts.row_messages's "consolidate" branch: new_memory/existing_memory each
    /// shaped as {"content","type":"episodic"} (existing_memory also carries its store id).
    /// </summary>
    public static string BuildConsolidationPrompt(string newContent, string existingMemoryId, string existingContent)
    {
        var payload = SortedJson(new Dictionary<string, object?>
        {
            ["operation"] = "consolidate",
            ["new_memory"] = new Dictionary<string, object?> { ["content"] = newContent, ["type"] = "episodic" },
            ["existing_memory"] = new Dictionary<string, object?>
            {
                ["id"] = existingMemoryId,
                ["content"] = existingContent,
                ["type"] = "episodic"
            }
        });
        return ChatMl(ConsolidationSystemInstruction, ConsolidationUserInstruction + payload);
    }

    // --- shared helpers --------------------------------------------------------------------------

    /// <summary>
    /// ChatML wrapper confirmed against psm-model/prod-memory/onnx-spike/foundry-local-spike/FoundrySpike/Program.cs
    /// and psm-model/prod-memory/onnx-runtime/v1/chat_template.jinja: system+user turns each end in
    /// `&lt;|im_end|&gt;`, followed by an open `&lt;|im_start|&gt;assistant` turn for the model to complete.
    /// </summary>
    private static string ChatMl(string system, string user) =>
        $"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        WriteIndented = false
    };

    /// <summary>
    /// Serializes a dictionary as compact JSON with keys sorted ordinally at every nesting level,
    /// matching Python's json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).
    /// </summary>
    private static string SortedJson(Dictionary<string, object?> fields) => SortedObject(fields).ToJsonString(JsonOptions);

    private static JsonNode? ToNode(object? value) => value switch
    {
        null => null,
        JsonNode node => node,
        string s => JsonValue.Create(s),
        int i => JsonValue.Create(i),
        long l => JsonValue.Create(l),
        double d => JsonValue.Create(d),
        bool b => JsonValue.Create(b),
        Dictionary<string, object?> dict => SortedObject(dict),
        IEnumerable<string> strings => new JsonArray(strings.Select(s => (JsonNode?)JsonValue.Create(s)).ToArray()),
        _ => JsonValue.Create(value.ToString())
    };

    private static JsonObject SortedObject(Dictionary<string, object?> fields)
    {
        var obj = new JsonObject();
        foreach (var key in fields.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            obj[key] = ToNode(fields[key]);
        }
        return obj;
    }
}
