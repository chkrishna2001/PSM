using System.Globalization;
using System.Text;

namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Atomic exclusive-create lock file guarding the "check health -> decide to spawn -> spawn"
/// sequence in <see cref="WarmHostClient.EnsureDaemonAsync"/>. This fixes a real bug in the original
/// TS design (src/psm-cli/src/daemon.ts's <c>ensureDaemon()</c> has no such lock, so two
/// near-simultaneous callers can both spawn a daemon): <see cref="FileMode.CreateNew"/> is atomic at
/// the filesystem level on both Windows and Linux, so this is a real cross-process mutex, not just an
/// in-process one.
/// </summary>
public static class WarmHostLock
{
    /// <summary>
    /// Attempts to acquire the lock at <paramref name="lockPath"/>. Returns an <see cref="IDisposable"/>
    /// that deletes the lock file when disposed, or null if someone else currently holds it and it
    /// isn't stale yet. <paramref name="now"/> is injectable so tests can simulate elapsed time
    /// without a real sleep.
    /// </summary>
    public static IDisposable? TryAcquire(string lockPath, TimeSpan staleThreshold, Func<DateTimeOffset>? now = null)
    {
        var clock = now ?? (() => DateTimeOffset.UtcNow);

        if (TryCreate(lockPath, clock, out var handle)) return handle;

        // Someone else already holds it -- is their lock stale?
        if (!IsStale(lockPath, staleThreshold, clock))
        {
            return null; // held, not stale -- caller should wait/poll for them.
        }

        // Stale (or unreadable/corrupt): reclaim by deleting and retrying CreateNew exactly once.
        // Another process could win the race between our delete and our retry -- that's fine, it just
        // means we lose this one attempt; we do not spin forever.
        try
        {
            File.Delete(lockPath);
        }
        catch
        {
            // Someone else may have already deleted/replaced/still holds it -- fall through to the one
            // bounded retry anyway; if it fails too we correctly return null below.
        }

        return TryCreate(lockPath, clock, out var retryHandle) ? retryHandle : null;
    }

    private static bool TryCreate(string lockPath, Func<DateTimeOffset> clock, out IDisposable? handle)
    {
        try
        {
            // FileShare.Read (not None): the handle stays open for the lifetime of the lock (per
            // design), but other processes must still be able to open it for reading so IsStale can
            // inspect the held timestamp while the lock is live.
            var stream = new FileStream(lockPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            var bytes = Encoding.UTF8.GetBytes(clock().ToString("O", CultureInfo.InvariantCulture));
            stream.Write(bytes, 0, bytes.Length);
            stream.Flush();
            handle = new LockHandle(stream, lockPath);
            return true;
        }
        catch (IOException)
        {
            handle = null;
            return false;
        }
    }

    private static bool IsStale(string lockPath, TimeSpan staleThreshold, Func<DateTimeOffset> clock)
    {
        var content = ReadLockFileTolerant(lockPath);
        if (content is null) return true; // unreadable/corrupt -- treat as stale, reclaim it.
        try
        {
            var timestamp = DateTimeOffset.Parse(content, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind);
            return clock() - timestamp >= staleThreshold;
        }
        catch
        {
            return true; // corrupt content -- treat as stale.
        }
    }

    private static string? ReadLockFileTolerant(string lockPath)
    {
        try
        {
            using var stream = new FileStream(lockPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }
        catch
        {
            return null;
        }
    }

    private sealed class LockHandle : IDisposable
    {
        private readonly FileStream _stream;
        private readonly string _path;
        private bool _disposed;

        public LockHandle(FileStream stream, string path)
        {
            _stream = stream;
            _path = path;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _stream.Dispose();
            try { File.Delete(_path); } catch { /* best effort */ }
        }
    }
}
