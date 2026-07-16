using System.Text.Json;

namespace PsmMemory.Core.Runtime;

/// <summary>
/// Downloads a folder from a public HuggingFace model repo via plain HTTP (the HF "tree" API to list
/// files + the "resolve" URL to fetch each one) -- no Python, no huggingface_hub dependency, so
/// PsmMemory.Core stays a pure .NET library. Used by <see cref="OnnxPsmRuntime.CreateAsync"/> to let
/// PSM bootstrap the ONNX model on first run instead of requiring the caller to have already run
/// psm-model/scripts/convert_adapters_onnx.py.
/// </summary>
public static class HfModelDownloader
{
    private const string ApiBase = "https://huggingface.co/api/models";
    private const string ResolveBase = "https://huggingface.co";

    private sealed record TreeEntry(string Type, string Path, long Size);

    /// <summary>
    /// Downloads every file under <paramref name="remotePath"/> in <paramref name="repoId"/> (HF's
    /// default "main" revision) into <paramref name="localDirectory"/>, preserving the relative
    /// directory structure. Each file is streamed to a ".part" temp file and renamed into place only
    /// after a full, successful download, so an interrupted run never leaves a corrupt file behind
    /// that <see cref="OnnxPsmRuntime"/>'s completeness check would mistake for a real one.
    /// </summary>
    public static async Task DownloadFolderAsync(
        string repoId,
        string remotePath,
        string localDirectory,
        HttpClient? httpClient = null,
        CancellationToken ct = default)
    {
        var client = httpClient ?? new HttpClient();
        var ownsClient = httpClient is null;
        try
        {
            var entries = await ListFilesAsync(client, repoId, remotePath, ct).ConfigureAwait(false);
            if (entries.Count == 0)
            {
                throw new InvalidOperationException(
                    $"No files found under '{remotePath}' in HF repo '{repoId}' -- check the repo id and path.");
            }

            Directory.CreateDirectory(localDirectory);
            var totalBytes = entries.Sum(e => e.Size);
            long downloaded = 0;

            Console.Error.WriteLine(
                $"downloading PSM model from huggingface.co/{repoId}/{remotePath} "
                + $"({entries.Count} files, {totalBytes / 1_000_000.0:F0} MB total)...");

            foreach (var entry in entries)
            {
                var relative = entry.Path[(remotePath.Length + 1)..];
                var localFile = System.IO.Path.Combine(localDirectory, relative.Replace('/', System.IO.Path.DirectorySeparatorChar));
                await DownloadFileAsync(client, repoId, entry.Path, localFile, ct).ConfigureAwait(false);
                downloaded += entry.Size;
                Console.Error.WriteLine($"  {relative} ({downloaded / 1_000_000.0:F0}/{totalBytes / 1_000_000.0:F0} MB)");
            }

            Console.Error.WriteLine("model download complete.");
        }
        finally
        {
            if (ownsClient) client.Dispose();
        }
    }

    private static async Task<List<TreeEntry>> ListFilesAsync(HttpClient client, string repoId, string remotePath, CancellationToken ct)
    {
        var url = $"{ApiBase}/{repoId}/tree/main/{remotePath}?recursive=true";
        using var response = await client.GetAsync(url, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(json);

        var result = new List<TreeEntry>();
        foreach (var item in doc.RootElement.EnumerateArray())
        {
            var type = item.GetProperty("type").GetString() ?? "";
            if (type != "file") continue;
            var path = item.GetProperty("path").GetString() ?? throw new InvalidOperationException("HF tree entry missing 'path'.");
            var size = item.TryGetProperty("size", out var sizeEl) ? sizeEl.GetInt64() : 0;
            result.Add(new TreeEntry(type, path, size));
        }
        return result;
    }

    private static async Task DownloadFileAsync(HttpClient client, string repoId, string remoteFilePath, string localFilePath, CancellationToken ct)
    {
        var dir = System.IO.Path.GetDirectoryName(localFilePath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var url = $"{ResolveBase}/{repoId}/resolve/main/{remoteFilePath}";
        var tempPath = localFilePath + ".part";

        using (var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false))
        {
            response.EnsureSuccessStatusCode();
            await using var httpStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            await using var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await httpStream.CopyToAsync(fileStream, ct).ConfigureAwait(false);
        }

        File.Move(tempPath, localFilePath, overwrite: true);
    }
}
