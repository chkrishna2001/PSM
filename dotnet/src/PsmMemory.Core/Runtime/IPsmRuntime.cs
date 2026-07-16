namespace PsmMemory.Core.Runtime;

/// <summary>
/// Which trained-adapter domain to use for a given call. Each domain has its own dedicated storage /
/// retrieval-plan / consolidation adapter set, sharing one base ONNX graph (same LoRA config across
/// all adapters in every domain is what makes this possible — see
/// psm-model/scripts/convert_adapters_onnx.py). <see cref="Coding"/> is trained on coding-agent
/// transcripts; <see cref="Conversational"/> is trained on personal/social memory content
/// (LoCoMo-style) and lets a user choose which kind of memory skill they want for a given call —
/// "some people can use this for normal work too."
/// </summary>
public enum PsmDomain
{
    Coding,
    Conversational,
}

/// <summary>
/// Replaces psm-core/src/psm-model-runtime.ts's single generic ModelRuntime.generateJson() with
/// three task-specific methods, one per trained LoRA adapter (storage / retrieval_plan /
/// consolidation), each parameterized by <see cref="PsmDomain"/>. The old TS interface had to be
/// generic because it shelled out to a single Python process; now that inference runs in-process via
/// ONNX Runtime GenAI, each adapter is a clearly separate capability and should be called through its
/// own method.
/// </summary>
public interface IPsmRuntime
{
    Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default);

    Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default);

    Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default);
}
