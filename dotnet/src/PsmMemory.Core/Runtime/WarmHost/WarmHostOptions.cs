namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Configuration shared by <see cref="WarmHostServer.RunAsync"/> (the process that loads the model
/// once and stays warm) and <see cref="WarmHostClient.EnsureDaemonAsync"/> (the short-lived caller
/// that reaches it). <see cref="StateDirectory"/> is caller-supplied (e.g. the directory containing
/// the memory db) rather than resolved from <see cref="Environment.SpecialFolder"/> inside this
/// class -- matching PsmMemory.Cli.InstallAgentCommand's <c>configDir</c> parameter pattern for
/// testability.
/// </summary>
public sealed class WarmHostOptions
{
    public required string ModelDir { get; init; }

    public required string StateDirectory { get; init; }

    public string Host { get; init; } = "127.0.0.1";

    public TimeSpan IdleTimeout { get; init; } = TimeSpan.FromMinutes(15);

    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromSeconds(60);

    public TimeSpan LockStaleThreshold { get; init; } = TimeSpan.FromSeconds(30);

    public string HfRepoId { get; init; } = LlamaSharpPsmRuntime.DefaultHfRepoId;

    public string StateFilePath => Path.Combine(StateDirectory, "daemon.json");

    public string LockFilePath => Path.Combine(StateDirectory, "daemon.lock");
}
