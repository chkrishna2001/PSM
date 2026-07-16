using System.Text.Json;

namespace PsmMemory.Cli;

internal static class CliRunner
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "-h" or "help")
        {
            Console.WriteLine(HelpText.Root);
            return 0;
        }

        var command = args[0];
        var rest = args[1..];

        return command switch
        {
            "remember" => await Commands.RunRememberAsync(rest).ConfigureAwait(false),
            "recall" => await Commands.RunRecallAsync(rest).ConfigureAwait(false),
            "context" => await Commands.RunContextAsync(rest).ConfigureAwait(false),
            "show" => Commands.RunShow(rest),
            "conflicts" => Commands.RunConflicts(rest),
            "init" or "migrate" => Commands.RunInit(rest),
            _ => throw new CliUsageException($"Unknown command '{command}'. Run 'psm-memory --help' for usage.")
        };
    }

    public static void PrintJson<T>(T value) => Console.WriteLine(JsonSerializer.Serialize(value, JsonOptions));
}

internal static class Defaults
{
    public const string DbPath = "user_memory.db";

    /// <summary>
    /// Best-effort default for --model-dir: walk up from the running executable's directory
    /// looking for psm-model/prod-memory/onnx-runtime/v1. Works out of the box when running from
    /// inside a checkout of this repo (e.g. dotnet build output under dotnet/src/PsmMemory.Cli/bin/...).
    /// If this tool is installed elsewhere (e.g. as a global dotnet tool), this will not find the
    /// repo and the caller must pass --model-dir explicitly -- EnsureModelDir below gives a clear
    /// error in that case rather than silently doing the wrong thing.
    /// </summary>
    public static string ResolveModelDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "psm-model", "prod-memory", "onnx-runtime", "v1");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return Path.Combine("psm-model", "prod-memory", "onnx-runtime", "v1");
    }

    /// <summary>
    /// No longer requires the directory to already contain a model -- OnnxPsmRuntime.CreateAsync
    /// downloads it from HuggingFace on first run if missing. Just ensures the path itself is usable
    /// as a download target.
    /// </summary>
    public static void EnsureModelDir(string modelDir)
    {
        Directory.CreateDirectory(modelDir);
    }
}
