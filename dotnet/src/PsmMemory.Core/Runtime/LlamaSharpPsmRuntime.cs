using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

namespace PsmMemory.Core.Runtime;

/// <summary>
/// In-process llama.cpp (GGUF) runtime for PSM's adapters (storage / retrieval_plan /
/// consolidation, per <see cref="PsmDomain"/>). Replaces an earlier onnxruntime-genai-backed
/// runtime: the swappable-LoRA dynamo-exported ONNX graph turned out to be both slow on CPU
/// (~10 tok/s for a 0.5B model -- a dynamic, Concat-based KV-cache recomputes/reallocates a
/// growing tensor every token) and unsalvageable via post-hoc quantization (Olive's RTN int4/int8
/// passes catastrophically broke output correctness, most likely because the official ORT GenAI LoRA
/// docs specify quantizing the base model BEFORE adapter extraction, not after). llama.cpp's GGUF
/// quantization is the mature, battle-tested alternative: validated on RunPod RTX A5000 at 83.3%
/// action-match on the full 419-case holdout-conversational-storage-cases.json gate (matching/
/// exceeding the 81.9% prior ONNX baseline) at 0.7s/case with full GPU offload, and ~40 tok/s even on
/// CPU with Q4_K_M quantization -- both a correctness and a ~10-40x speed win over the ONNX path.
///
/// Architecture: one <see cref="LLamaWeights"/> (base model, loaded once) + one persistent
/// <see cref="LLamaContext"/> (created once) + N <see cref="LoraAdapter"/> (one per available
/// domain/task, each loaded once via <see cref="LLamaWeights.LoadLoraFromFile"/>). Each generation
/// call activates the right adapter via <see cref="SafeLLamaContextHandle.SetLoraAdapters"/> and
/// clears the KV-cache via <see cref="SafeLLamaContextHandle.MemoryClear"/> for a clean, independent
/// single-turn generation -- validated directly against llama.cpp's low-level C API (swap between
/// two different adapters on the same loaded context with zero reload, output changes correctly per
/// adapter, and swapping back reproduces the identical original output).
///
/// GPU/NPU/CPU selection needs no OS-specific try/catch fallback ladder: <see cref="ModelParams.GpuLayerCount"/>
/// = -1 asks llama.cpp to offload every layer it can, and it transparently falls back to partial or
/// full CPU for whatever doesn't fit (or entirely, if no GPU backend is bundled/available) -- this is
/// llama.cpp's own built-in behavior, not something this class has to implement.
/// </summary>
public sealed class LlamaSharpPsmRuntime : IPsmRuntime, IDisposable
{
    /// <summary>
    /// Default model directory, relative to the repo root. Callers must resolve this themselves and
    /// pass an absolute or otherwise-resolved path to the constructor.
    /// </summary>
    public const string DefaultRelativeModelDirectory = "psm-model/prod-memory/gguf-runtime/v1";

    /// <summary>HF repo <see cref="CreateAsync"/> downloads from when the local model directory is missing/incomplete.</summary>
    public const string DefaultHfRepoId = "chkrishna2001/psm-memory-qwen0.5b";

    /// <summary>Path within <see cref="DefaultHfRepoId"/> containing the GGUF base model + LoRA adapters.</summary>
    public const string DefaultHfRemotePath = "gguf";

    private const string BaseModelFileName = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

    private const uint ContextSize = 4096;
    private const int MaxNewTokens = 768;

    private enum AdapterTask
    {
        Storage,
        RetrievalPlan,
        Consolidation,
    }

    private readonly LLamaWeights _weights;
    private readonly LLamaContext _context;
    private readonly LLamaBatch _batch = new();
    private readonly Dictionary<string, LoraAdapter> _adapters = new();
    private readonly SemaphoreSlim _generationLock = new(1, 1);
    private readonly HashSet<PsmDomain> _availableDomains = new();
    private bool _disposed;

    /// <summary>
    /// Ensures the model is present at <paramref name="modelDirectory"/> — downloading it from
    /// <paramref name="hfRepoId"/> (public repo, no auth needed) first if the Coding domain's files
    /// aren't already there — then constructs the runtime. Use the plain constructor instead when
    /// the model directory is already known-good (e.g. in tests) and a network call is undesirable.
    /// </summary>
    public static async Task<LlamaSharpPsmRuntime> CreateAsync(
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
        return new LlamaSharpPsmRuntime(modelDirectory);
    }

    private static bool HasCodingAdapters(string modelDirectory) =>
        File.Exists(Path.Combine(modelDirectory, BaseModelFileName))
        && File.Exists(Path.Combine(modelDirectory, "adapters", "storage-lora-f16.gguf"))
        && File.Exists(Path.Combine(modelDirectory, "adapters", "retrieval_plan-lora-f16.gguf"))
        && File.Exists(Path.Combine(modelDirectory, "adapters", "consolidation-lora-f16.gguf"));

    public LlamaSharpPsmRuntime(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
            throw new ArgumentException("modelDirectory must be a non-empty path.", nameof(modelDirectory));
        if (!Directory.Exists(modelDirectory))
            throw new DirectoryNotFoundException($"PSM GGUF model directory not found: {modelDirectory}");

        var modelPath = Path.Combine(modelDirectory, BaseModelFileName);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Base GGUF model not found at {modelPath}", modelPath);

        var modelParams = new ModelParams(modelPath)
        {
            ContextSize = ContextSize,
            GpuLayerCount = -1,
        };
        _weights = LLamaWeights.LoadFromFile(modelParams);
        _context = _weights.CreateContext(modelParams);

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
                + "storage/retrieval_plan/consolidation-lora-f16.gguf files, produced by "
                + "psm-model/scripts/convert_adapters_gguf.py.");
        }
    }

    /// <summary>Domains whose adapters were actually found and loaded at construction time.</summary>
    public IReadOnlySet<PsmDomain> AvailableDomains => _availableDomains;

    private bool TryLoadDomainAdapters(string adaptersDir, PsmDomain domain)
    {
        var prefix = AdapterFilePrefix(domain);
        var paths = new[]
        {
            (AdapterTask.Storage, Path.Combine(adaptersDir, $"{prefix}storage-lora-f16.gguf")),
            (AdapterTask.RetrievalPlan, Path.Combine(adaptersDir, $"{prefix}retrieval_plan-lora-f16.gguf")),
            (AdapterTask.Consolidation, Path.Combine(adaptersDir, $"{prefix}consolidation-lora-f16.gguf")),
        };

        // All three must be present for a domain to count as available — a partial set is not
        // usable and would silently break the other two tasks.
        if (paths.Any(p => !File.Exists(p.Item2))) return false;

        foreach (var (task, path) in paths)
        {
            _adapters[AdapterName(domain, task)] = _weights.NativeHandle.LoadLoraFromFile(path);
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
                + "psm-model/scripts/convert_adapters_gguf.py before requesting this domain.");
        }
    }

    public Task<string> GenerateStorageDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
    {
        EnsureDomainAvailable(domain);
        return GenerateAsync(prompt, AdapterName(domain, AdapterTask.Storage), earlyStopOnIgnore: true, ct);
    }

    public Task<string> GenerateRecallPlanAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
    {
        EnsureDomainAvailable(domain);
        return GenerateAsync(prompt, AdapterName(domain, AdapterTask.RetrievalPlan), earlyStopOnIgnore: false, ct);
    }

    public Task<string> GenerateConsolidationDecisionAsync(string prompt, PsmDomain domain = PsmDomain.Coding, CancellationToken ct = default)
    {
        EnsureDomainAvailable(domain);
        return GenerateAsync(prompt, AdapterName(domain, AdapterTask.Consolidation), earlyStopOnIgnore: false, ct);
    }

    private async Task<string> GenerateAsync(string prompt, string adapterName, bool earlyStopOnIgnore, CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // One shared LLamaContext across all calls (that's the whole point of the hot-swap
        // architecture) — serialize generation so callers can safely share one runtime instance
        // across concurrent PsmService calls.
        await _generationLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            return await Generate(prompt, adapterName, earlyStopOnIgnore, ct).ConfigureAwait(false);
        }
        finally
        {
            _generationLock.Release();
        }
    }

    // Once the model commits to action=ignore, the rest of the object (memory=null, empty
    // facts/indexables) is fixed boilerplate nobody reads — force-close the JSON right there
    // instead of generating it out token by token.
    private const string IgnoreActionMarker = "\"action\":\"ignore";
    private const string IgnoreJsonClosingSuffix = "\",\"memory\":null,\"facts\":[],\"indexables\":[]}";

    private async Task<string> Generate(string prompt, string adapterName, bool earlyStopOnIgnore, CancellationToken ct)
    {
        // Activate only the requested adapter on the shared, persistent context (validated: swapping
        // adapters this way on the same context produces correct, adapter-specific output with zero
        // base-model reload), then clear the KV-cache so this generation starts clean, independent of
        // whatever the previous call (possibly a different adapter/task) left behind.
        _context.NativeHandle.SetLoraAdapters((_adapters[adapterName], 1.0f));
        _context.NativeHandle.MemoryClear();

        var sampler = new GreedySamplingPipeline();
        var decoder = new StreamingTokenDecoder(_context);

        var tokens = _context.Tokenize(prompt, addBos: true, special: true).ToList();
        var (decodeResult, _, past) = await _context.DecodeAsync(tokens, LLamaSeqId.Zero, _batch, 0, ct).ConfigureAwait(false);
        if (decodeResult != DecodeResult.Ok)
            throw new InvalidOperationException($"llama.cpp prompt decode failed: {decodeResult}");
        var nPast = past;

        var textSoFar = new StringBuilder();
        var generated = 0;
        var ignoreDetected = false;
        while (generated < MaxNewTokens)
        {
            var id = sampler.Sample(_context.NativeHandle, _batch.TokenCount - 1);
            if (id.IsEndOfGeneration(_weights.Vocab)) break;

            decoder.Add(id);
            textSoFar.Append(decoder.Read());
            generated++;

            if (earlyStopOnIgnore && generated >= 6 && textSoFar.ToString().Contains(IgnoreActionMarker, StringComparison.Ordinal))
            {
                ignoreDetected = true;
                break;
            }

            _batch.Clear();
            _batch.Add(id, nPast++, LLamaSeqId.Zero, true);
            var next = await _context.DecodeAsync(_batch, ct).ConfigureAwait(false);
            if (next != DecodeResult.Ok)
                throw new InvalidOperationException($"llama.cpp decode failed: {next}");
        }

        var text = textSoFar.ToString();
        if (ignoreDetected)
        {
            var markerIndex = text.IndexOf(IgnoreActionMarker, StringComparison.Ordinal);
            if (markerIndex >= 0)
            {
                var prefixEnd = markerIndex + IgnoreActionMarker.Length;
                text = text[..prefixEnd] + IgnoreJsonClosingSuffix;
            }
        }

        return text;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _generationLock.Dispose();
        foreach (var adapter in _adapters.Values) adapter.Unload();
        _context.Dispose();
        _weights.Dispose();
    }
}
