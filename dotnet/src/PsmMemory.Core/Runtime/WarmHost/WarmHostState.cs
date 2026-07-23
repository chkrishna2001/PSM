using System.Text.Json;

namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Ported from src/psm-cli/src/daemon.ts's <c>{pid, host, port, startedAt, lastSeenAt}</c> state file
/// (daemon.json). Rewritten on every request the warm host serves -- that rewrite of
/// <see cref="LastSeenAt"/> IS the sliding-expiration mechanism (see <see cref="WarmHostServer"/>).
/// </summary>
public sealed class DaemonState
{
    public int Pid { get; set; }
    public required string Host { get; set; }
    public int Port { get; set; }
    public DateTimeOffset StartedAt { get; set; }
    public DateTimeOffset LastSeenAt { get; set; }
}

/// <summary>
/// Tolerant reader / atomic writer for daemon.json. Unlike TS's plain <c>writeFileSync</c> (which can
/// in theory be interrupted mid-write), <see cref="Write"/> writes to a temp file in the same
/// directory and then <see cref="File.Move(string, string, bool)"/>s it into place -- an atomic rename
/// on both Windows and Linux for same-volume moves -- so a concurrent reader never observes a
/// half-written file.
/// </summary>
public static class WarmHostState
{
    /// <summary>Never throws: a missing file, corrupt/partial JSON, or a wrong shape (e.g. missing the
    /// required <see cref="DaemonState.Host"/> field) all mean "no usable state" to a caller like
    /// <see cref="WarmHostClient.EnsureDaemonAsync"/>, which just treats that as "no daemon running
    /// yet".</summary>
    public static DaemonState? Read(string path)
    {
        try
        {
            if (!File.Exists(path)) return null;
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<DaemonState>(json, WarmHostJson.Options);
        }
        catch
        {
            return null;
        }
    }

    public static void Write(string path, DaemonState state)
    {
        var json = JsonSerializer.Serialize(state, WarmHostJson.Options);
        var tempPath = path + ".tmp";
        File.WriteAllText(tempPath, json);
        File.Move(tempPath, path, overwrite: true);
    }
}
