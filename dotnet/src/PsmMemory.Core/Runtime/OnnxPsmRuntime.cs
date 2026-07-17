using Microsoft.ML.OnnxRuntimeGenAI;

namespace PsmMemory.Core.Runtime;

/// <summary>
/// In-process ONNX Runtime GenAI runtime for PSM's adapters (storage / retrieval_plan /
/// consolidation, per <see cref="PsmDomain"/>), replacing psm-core/src/psm-model-runtime.ts's
/// Python-subprocess mechanism entirely. Ported from the validated pattern in
/// psm-model/prod-memory/onnx-spike/foundry-local-spike/FoundrySpike/Program.cs (0.84 gate-score
/// parity with the original PyTorch model): base model + tokenizer are loaded once, every available
/// domain's LoRA adapters are loaded once via <see cref="Adapters.LoadAdapter"/>, and each generation
/// call creates its own GeneratorParams/Generator, activates the right adapter via
/// <see cref="Generator.SetActiveAdapter"/>, and greedy-decodes (do_sample=false).
///
/// Only <see cref="PsmDomain.Coding"/> adapters exist today. <see cref="PsmDomain.Conversational"/>
/// adapters are trained separately (LoCoMo-style personal/social memory data) and are loaded
/// automatically once their .onnx_adapter files exist in the model directory's adapters/ folder,
/// named "conversational_storage" / "conversational_retrieval_plan" / "conversational_consolidation"
/// — no code change needed here when that lands, only new files on disk. Requesting a domain whose
/// adapters aren't present throws a clear error rather than silently falling back to another domain.
/// </summary>
public sealed class OnnxPsmRuntime : IPsmRuntime, IDisposable
{
    /// <summary>
    /// Default model directory, relative to the repo root. Callers must resolve this themselves
    /// (e.g. against their own notion of "repo root" or an installed model location) and pass an
    /// absolute or otherwise-resolved path to the constructor — this SDK never hardcodes a
    /// machine-specific absolute path.
    /// </summary>
    public const string DefaultRelativeModelDirectory = "psm-model/prod-memory/onnx-runtime/v2";

    /// <summary>HF repo <see cref="CreateAsync"/> downloads from when the local model directory is missing/incomplete.</summary>
    public const string DefaultHfRepoId = "chkrishna2001/psm-memory-qwen0.5b";

    /// <summary>Path within <see cref="DefaultHfRepoId"/> containing the ONNX model + adapters.</summary>
    public const string DefaultHfRemotePath = "onnx";

    // Matches FoundrySpike's validated search options exactly.
    private const int MaxLength = 4096;
    private const int MaxNewTokens = 768;

    private enum AdapterTask
    {
        Storage,
        RetrievalPlan,
        Consolidation,
    }

    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly Adapters _adapters;
    private readonly SemaphoreSlim _generationLock = new(1, 1);
    private readonly HashSet<PsmDomain> _availableDomains = new();
    private bool _disposed;

    /// <summary>
    /// Ensures the model is present at <paramref name="modelDirectory"/> — downloading it from
    /// <paramref name="hfRepoId"/> (public repo, no auth needed) via <see cref="HfModelDownloader"/>
    /// first if the Coding domain's files aren't already there — then constructs the runtime. This is
    /// what lets a fresh PSM install bootstrap itself without the caller having run
    /// psm-model/scripts/convert_adapters_onnx.py locally. Use the plain constructor instead when the
    /// model directory is already known-good (e.g. in tests) and a network call is undesirable.
    /// </summary>
    public static async Task<OnnxPsmRuntime> CreateAsync(
        string modelDirectory,
        string hfRepoId = DefaultHfRepoId,
        HttpClient? httpClient = null,
        CancellationToken ct = default)
    {
        if (!HasCodingAdapters(modelDirectory))
        {
            await HfModelDownloader.DownloadFolderAsync(hfRepoId, DefaultHfRemotePath, modelDirectory, httpClient, ct)
                .ConfigureAwait(false);
        }
        return new OnnxPsmRuntime(modelDirectory);
    }

    private static bool HasCodingAdapters(string modelDirectory) =>
        File.Exists(Path.Combine(modelDirectory, "model.onnx"))
        && File.Exists(Path.Combine(modelDirectory, "genai_config.json"))
        && File.Exists(Path.Combine(modelDirectory, "adapters", "storage.onnx_adapter"))
        && File.Exists(Path.Combine(modelDirectory, "adapters", "retrieval_plan.onnx_adapter"))
        && File.Exists(Path.Combine(modelDirectory, "adapters", "consolidation.onnx_adapter"));

    public OnnxPsmRuntime(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
            throw new ArgumentException("modelDirectory must be a non-empty path.", nameof(modelDirectory));
        if (!Directory.Exists(modelDirectory))
            throw new DirectoryNotFoundException($"PSM ONNX model directory not found: {modelDirectory}");

        _model = new Model(modelDirectory);
        _tokenizer = new Tokenizer(_model);
        _adapters = new Adapters(_model);

        var adaptersDir = Path.Combine(modelDirectory, "adapters");
        foreach (var domain in Enum.GetValues<PsmDomain>())
        {
            if (!TryLoadDomainAdapters(adaptersDir, domain)) continue;
            _availableDomains.Add(domain);
        }

        if (_availableDomains.Count == 0)
        {
            throw new InvalidOperationException(
                $"No adapter set found under {adaptersDir} — expected at least the Coding domain's "
                + "storage/retrieval_plan/consolidation.onnx_adapter files, produced by "
                + "psm-model/scripts/convert_adapters_onnx.py.");
        }
    }

    /// <summary>Domains whose adapters were actually found and loaded at construction time.</summary>
    public IReadOnlySet<PsmDomain> AvailableDomains => _availableDomains;

    private bool TryLoadDomainAdapters(string adaptersDir, PsmDomain domain)
    {
        var prefix = AdapterFilePrefix(domain);
        var paths = new[]
        {
            (AdapterTask.Storage, Path.Combine(adaptersDir, $"{prefix}storage.onnx_adapter")),
            (AdapterTask.RetrievalPlan, Path.Combine(adaptersDir, $"{prefix}retrieval_plan.onnx_adapter")),
            (AdapterTask.Consolidation, Path.Combine(adaptersDir, $"{prefix}consolidation.onnx_adapter")),
        };

        // All three must be present for a domain to count as available — a partial set (e.g. only
        // storage trained so far) is not usable and would silently break the other two tasks.
        if (paths.Any(p => !File.Exists(p.Item2))) return false;

        foreach (var (task, path) in paths)
        {
            _adapters.LoadAdapter(path, AdapterName(domain, task));
        }
        return true;
    }

    private static string AdapterFilePrefix(PsmDomain domain) => domain switch
    {
        PsmDomain.Coding => "",
        PsmDomain.Conversational => "conversational_",
        _ => throw new ArgumentOutOfRangeException(nameof(domain), domain, null),
    };

    private static string AdapterName(PsmDomain domain, AdapterTask task) => $"{AdapterFilePrefix(domain)}{task switch
    {
        AdapterTask.Storage => "storage",
        AdapterTask.RetrievalPlan => "retrieval_plan",
        AdapterTask.Consolidation => "consolidation",
        _ => throw new ArgumentOutOfRangeException(nameof(task), task, null),
    }}";

    private void EnsureDomainAvailable(PsmDomain domain)
    {
        if (!_availableDomains.Contains(domain))
        {
            throw new InvalidOperationException(
                $"PsmDomain.{domain} has no trained adapters loaded (available: "
                + $"{string.Join(", ", _availableDomains)}). Train and convert its adapters via "
                + "psm-model/scripts/convert_adapters_onnx.py before requesting this domain.");
        }
    }

    public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
    {
        EnsureDomainAvailable(domain);
        return GenerateAsync(prompt, AdapterName(domain, AdapterTask.Storage), ct);
    }

    public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
    {
        EnsureDomainAvailable(domain);
        return GenerateAsync(prompt, AdapterName(domain, AdapterTask.RetrievalPlan), ct);
    }

    public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
    {
        EnsureDomainAvailable(domain);
        return GenerateAsync(prompt, AdapterName(domain, AdapterTask.Consolidation), ct);
    }

    private async Task<string> GenerateAsync(string prompt, string adapterName, CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // onnxruntime-genai's Model/Generator are not documented as safe for concurrent Run() calls
        // from multiple threads; serialize generation so callers can safely share one runtime
        // instance across concurrent PsmService calls.
        await _generationLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => Generate(prompt, adapterName), ct).ConfigureAwait(false);
        }
        finally
        {
            _generationLock.Release();
        }
    }

    private string Generate(string prompt, string adapterName)
    {
        using var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("do_sample", false);
        generatorParams.SetSearchOption("max_length", MaxLength);

        using var generator = new Generator(_model, generatorParams);
        generator.SetActiveAdapter(_adapters, adapterName);

        var promptSequences = _tokenizer.Encode(prompt);
        generator.AppendTokenSequences(promptSequences);
        var promptTokenCount = generator.TokenCount();

        var generated = 0;
        while (!generator.IsDone() && generated < MaxNewTokens)
        {
            generator.GenerateNextToken();
            generated++;
        }

        var fullSequence = generator.GetSequence(0);
        var newTokens = fullSequence.Slice((int)promptTokenCount);
        // Decode the full accumulated token-id list in one call (not token-by-token appended
        // strings) to avoid mangling multi-byte UTF-8 characters -- same fix applied in
        // psm-model/scripts/convert_adapters_onnx.py's Python smoke test this session.
        return _tokenizer.Decode(newTokens);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _generationLock.Dispose();
        _adapters.Dispose();
        _tokenizer.Dispose();
        _model.Dispose();
    }
}
