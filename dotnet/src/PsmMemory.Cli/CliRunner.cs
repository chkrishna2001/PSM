using System.Text.Json;
using PsmMemory.Core.Runtime.WarmHost;

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
            "serve" => await Commands.RunServeAsync(rest).ConfigureAwait(false),
            "show" => Commands.RunShow(rest),
            "conflicts" => Commands.RunConflicts(rest),
            "init" or "migrate" => Commands.RunInit(rest),
            "enqueue-remember" => Commands.RunEnqueueRemember(rest),
            "drain-queue" => await Commands.RunDrainQueueAsync(rest).ConfigureAwait(false),
            "hook" => await HookCommands.RunAsync(rest).ConfigureAwait(false),
            "install-agent" => InstallAgentCommand.Run(rest),
            "daemon-run" => await Commands.RunDaemonRunAsync(rest).ConfigureAwait(false),
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
    /// looking for PsmMemory.Core.Runtime.LlamaSharpPsmRuntime.DefaultRelativeModelDirectory. Works
    /// out of the box when running from inside a checkout of this repo (e.g. dotnet build output
    /// under dotnet/src/PsmMemory.Cli/bin/...). If this tool is installed elsewhere (e.g. as a
    /// global dotnet tool), this will not find the repo and the caller must pass --model-dir
    /// explicitly -- EnsureModelDir below gives a clear error in that case rather than silently
    /// doing the wrong thing.
    /// </summary>
    public static string ResolveModelDir() => PsmMemory.Core.Runtime.ModelDirResolver.ResolveFromBaseDirectory(
        AppContext.BaseDirectory,
        PsmMemory.Core.Runtime.LlamaSharpPsmRuntime.DefaultRelativeModelDirectory);

    /// <summary>
    /// No longer requires the directory to already contain a model -- LlamaSharpPsmRuntime.CreateAsync
    /// downloads it from HuggingFace on first run if missing. Just ensures the path itself is usable
    /// as a download target.
    /// </summary>
    public static void EnsureModelDir(string modelDir)
    {
        Directory.CreateDirectory(modelDir);
    }

    /// <summary>
    /// Whether this call should try the warm-host daemon before falling back to a direct local
    /// model load. Per-invocation --daemon/--no-daemon always win over the env var; with neither
    /// flag passed, PSM_MEMORY_DAEMON="on" (case-insensitive) enables it -- default is off, since a
    /// subsystem that spawns detached processes should be opt-in until proven.
    /// </summary>
    public static bool DaemonEnabled(ArgParser parsed)
    {
        if (parsed.HasFlag("no-daemon")) return false;
        if (parsed.HasFlag("daemon")) return true;
        return string.Equals(Environment.GetEnvironmentVariable("PSM_MEMORY_DAEMON"), "on", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Colocated with --db, matching the existing hook-audit-log convention of placing
    /// PSM-managed sidecar files next to the memory database rather than in a separate config root.</summary>
    public static string ResolveDaemonStateDirectory(string dbPath)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(dbPath));
        return string.IsNullOrEmpty(dir) ? Directory.GetCurrentDirectory() : dir;
    }

    /// <summary>Null means "daemon disabled for this call" -- callers pass this straight into
    /// PsmRuntimeAcquisition.AcquireAsync, which treats null the same way.</summary>
    public static WarmHostOptions? BuildWarmHostOptions(ArgParser parsed, string modelDir, string dbPath) =>
        DaemonEnabled(parsed)
            ? new WarmHostOptions { ModelDir = modelDir, StateDirectory = ResolveDaemonStateDirectory(dbPath) }
            : null;

    /// <summary>Args used to re-launch this same executable as a detached warm-host process (see
    /// WarmHostClient.SpawnDetached) -- deliberately built here rather than inside Core, so Core
    /// doesn't need to know "daemon-run" is a CLI-specific command name.</summary>
    public static string[] DaemonProcessArgs(string modelDir, string stateDirectory) =>
        new[] { "daemon-run", "--model-dir", modelDir, "--state-dir", stateDirectory };
}
