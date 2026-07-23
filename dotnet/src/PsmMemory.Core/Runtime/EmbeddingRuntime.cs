using LLama;
using LLama.Common;
using LLama.Native;

namespace PsmMemory.Core.Runtime;

/// <summary>
/// Ported from psm-core/src/embeddings.ts's <c>TransformersEmbeddingRuntime</c> (the
/// <c>EmbeddingRuntime</c> interface in psm-core/src/types.ts is the direct analogue of this
/// interface). Domain-agnostic: embeddings are computed from raw memory content text and never
/// touch a <see cref="PsmDomain"/>-specific adapter, so this same runtime backs vector recall for
/// both PsmDomain.Coding and PsmDomain.Conversational callers of PsmService.
/// </summary>
public interface IEmbeddingRuntime
{
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
}

/// <summary>
/// GGUF/LLamaSharp-backed embedding runtime. Uses a llama.cpp-compatible port of
/// Xenova/all-MiniLM-L6-v2 (the same model psm-core's TypeScript original used via
/// transformers.js) for parity of retrieval quality with the pre-port behavior -- see
/// psm-core/src/embeddings.ts's <c>defaultEmbeddingModel</c>. Mean-pooled, L2-normalized output,
/// matching transformers.js's <c>{ pooling: "mean", normalize: true }</c> call.
/// </summary>
public sealed class LlamaSharpEmbeddingRuntime : IEmbeddingRuntime, IDisposable
{
    /// <summary>
    /// Model name recorded in <c>memory_embeddings.model</c> -- callers must use the same model
    /// name for both writing (PsmService.RememberAsync) and reading (PlanAndRankAsync) embeddings,
    /// or stored embeddings silently won't be found. Ported from psm-core's constructor arg
    /// `embeddings: { model, runtime }`, where `model` played the same role.
    /// </summary>
    public const string ModelName = "all-MiniLM-L6-v2";

    /// <summary>HF repo <see cref="CreateAsync"/> downloads from when the local embedding model is missing.</summary>
    public const string DefaultHfRepoId = "chkrishna2001/psm-memory-qwen0.5b";

    /// <summary>Path within <see cref="DefaultHfRepoId"/> containing the embedding GGUF.</summary>
    public const string DefaultHfRemotePath = "gguf/embedding";

    private const string ModelFileName = "all-MiniLM-L6-v2-Q8_0.gguf";

    // all-MiniLM-L6-v2's own trained max sequence length.
    private const uint ContextSize = 512;

    private readonly LLamaWeights _weights;
    private readonly LLamaEmbedder _embedder;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private bool _disposed;

    /// <summary>
    /// Ensures the embedding model is present under <paramref name="modelDirectory"/>/embedding --
    /// downloading it from <paramref name="hfRepoId"/> first if missing -- then constructs the
    /// runtime. Mirrors <see cref="LlamaSharpPsmRuntime.CreateAsync"/>'s bootstrap pattern.
    /// </summary>
    public static async Task<LlamaSharpEmbeddingRuntime> CreateAsync(
        string modelDirectory,
        string hfRepoId = DefaultHfRepoId,
        HttpClient? httpClient = null,
        CancellationToken ct = default)
    {
        var embeddingDir = Path.Combine(modelDirectory, "embedding");
        if (!File.Exists(Path.Combine(embeddingDir, ModelFileName)))
        {
            await HfModelDownloader.DownloadFolderAsync(hfRepoId, DefaultHfRemotePath, embeddingDir, httpClient, ct)
                .ConfigureAwait(false);
        }
        return new LlamaSharpEmbeddingRuntime(modelDirectory);
    }

    public LlamaSharpEmbeddingRuntime(string modelDirectory)
    {
        var modelPath = Path.Combine(modelDirectory, "embedding", ModelFileName);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Embedding GGUF model not found at {modelPath}", modelPath);

        var modelParams = new ModelParams(modelPath)
        {
            ContextSize = ContextSize,
            Embeddings = true,
            PoolingType = LLamaPoolingType.Mean,
            GpuLayerCount = -1,
        };
        _weights = LLamaWeights.LoadFromFile(modelParams);
        _embedder = new LLamaEmbedder(_weights, modelParams);
    }

    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // LLamaEmbedder shares llama.cpp state the same way LlamaSharpPsmRuntime's generation
        // context does -- serialize calls so one runtime instance is safe to share across
        // concurrent PsmService callers.
        await _lock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var results = await _embedder.GetEmbeddings(text, ct).ConfigureAwait(false);
            if (results.Count == 0)
                throw new InvalidOperationException("llama.cpp returned no embedding for the given text.");
            return Normalize(results[0]);
        }
        finally
        {
            _lock.Release();
        }
    }

    private static float[] Normalize(float[] vector)
    {
        double sumSquares = 0;
        foreach (var value in vector) sumSquares += (double)value * value;
        var norm = Math.Sqrt(sumSquares);
        if (norm < 1e-9) return vector;
        var normalized = new float[vector.Length];
        for (var i = 0; i < vector.Length; i++) normalized[i] = (float)(vector[i] / norm);
        return normalized;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _lock.Dispose();
        _embedder.Dispose();
        _weights.Dispose();
    }
}
