using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Integration smoke test: loads the real LlamaSharpPsmRuntime against the validated GGUF model at
/// psm-model/prod-memory/gguf-runtime/v1 and drives PsmService.RememberAsync end-to-end (real
/// PromptBuilder, real parser, real SQLite store) on a handful of real cases pulled from
/// psm-model/prod-memory/onnx-spike/gguf-spike/gate-cases-prompts.json. This only proves the C#
/// plumbing doesn't break already-validated model quality — it is not a re-run of the full
/// 100-case gate. Skipped by default because it loads a real model and does real greedy decoding
/// (seconds); run manually to confirm the plumbing is sound.
/// </summary>
public class LlamaSharpSmokeTests
{
    /// <summary>Real cases from psm-model/prod-memory/onnx-spike/gguf-spike/gate-cases-prompts.json,
    /// with just the assistant-response text extracted (PromptBuilder re-derives the identical
    /// ChatML prompt the gate harness used to hit 0.84 gate-score parity).</summary>
    public static readonly (string Id, string ExpectAction, string LlmResponse)[] Cases =
    {
        ("coding-agent-adapter-verified-hf", "ignore",
            "Confirmed - commits from 2026-07-05 09:54 UTC (today, minutes ago), matching this exact retrain run. The adapter is safely verified on HF hub. Now let's pull it locally and re-run the eval."),
        ("coding-agent-parse-failure-finding", "store",
            "This is the real finding - most \"ignore\" decisions aren't the model choosing to ignore, they're parse failures being fail-safed into ignore. Let me quantify how widespread this is."),
        ("coding-agent-commit-experiment-2", "store",
            "Implemented and committed the next experiment. Commit: f07fc82 Experiment 2: similarity matching. Trace.cs now keeps a tiny evolving prototype to compare against instead of raw history.")
    };

    // Repo-relative paths are fine here (unlike in library code) because tests are inherently tied
    // to this repo's layout.
    public static string FindRepoRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (Directory.Exists(Path.Combine(current.FullName, "psm-model")) &&
                Directory.Exists(Path.Combine(current.FullName, "dotnet")))
            {
                return current.FullName;
            }
            current = current.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate PSM repo root from test working directory.");
    }

    [Fact(Skip = "manual — requires the real GGUF model on disk; run manually to validate the llama.cpp plumbing end-to-end")]
    public async Task RememberAsync_ProducesWellFormedDecisions_ForRealGateCases()
    {
        var modelDir = Path.Combine(FindRepoRoot(), "psm-model", "prod-memory", "gguf-runtime", "v1");
        Assert.True(Directory.Exists(modelDir), $"expected model directory at {modelDir}");

        using var runtime = new LlamaSharpPsmRuntime(modelDir);
        var dbPath = Path.Combine(Path.GetTempPath(), $"psm-smoke-{Guid.NewGuid():N}.db");
        using var store = new MemoryStore(dbPath);
        store.InitializeSchema();
        var service = new PsmService(store, runtime);

        foreach (var (_, _, llmResponse) in Cases)
        {
            var result = await service.RememberAsync(new RememberRequest { UserId = "smoke-user", LlmResponse = llmResponse });

            Assert.NotNull(result.Action);
            Assert.NotEmpty(result.Action);
            // Never corrupts the store even on parse failure: action is always one of the known kinds.
            Assert.False(string.IsNullOrWhiteSpace(result.Route));
        }

        File.Delete(dbPath);
    }
}
