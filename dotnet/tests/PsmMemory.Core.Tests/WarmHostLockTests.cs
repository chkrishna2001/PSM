using PsmMemory.Core.Runtime.WarmHost;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Covers <see cref="WarmHostLock"/>'s atomic exclusive-create lock file, including the
/// bounded stale-reclaim retry that fixes a real bug in the original TS design (no locking at all
/// around concurrent auto-spawn -- see src/psm-cli/src/daemon.ts's <c>ensureDaemon()</c>).
/// </summary>
public class WarmHostLockTests
{
    private static readonly TimeSpan StaleThreshold = TimeSpan.FromSeconds(30);

    private static string TempLockPath() => Path.Combine(Path.GetTempPath(), $"psm-warmhost-lock-{Guid.NewGuid():N}.lock");

    [Fact]
    public void TryAcquire_NoExistingLock_Succeeds()
    {
        var path = TempLockPath();
        try
        {
            using var handle = WarmHostLock.TryAcquire(path, StaleThreshold);
            Assert.NotNull(handle);
            Assert.True(File.Exists(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TryAcquire_AlreadyHeldAndNotStale_ReturnsNull()
    {
        var path = TempLockPath();
        try
        {
            using var first = WarmHostLock.TryAcquire(path, StaleThreshold);
            Assert.NotNull(first);

            var second = WarmHostLock.TryAcquire(path, StaleThreshold);
            Assert.Null(second);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TryAcquire_StaleLockFromACrashedHolder_CanBeReclaimed()
    {
        var path = TempLockPath();
        try
        {
            // Simulate a crashed lock-holder: an orphaned lock file exists on disk (its writer's
            // process is gone, so no live handle blocks a delete) with an old timestamp, well past
            // staleThreshold relative to "now".
            var oldTimestamp = DateTimeOffset.UtcNow.AddMinutes(-5);
            File.WriteAllText(path, oldTimestamp.ToString("O"));

            var now = DateTimeOffset.UtcNow;
            using var reclaimed = WarmHostLock.TryAcquire(path, StaleThreshold, () => now);

            Assert.NotNull(reclaimed);
            Assert.True(File.Exists(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TryAcquire_NotYetStale_DoesNotReclaim()
    {
        var path = TempLockPath();
        try
        {
            var recentTimestamp = DateTimeOffset.UtcNow.AddSeconds(-5);
            File.WriteAllText(path, recentTimestamp.ToString("O"));

            var now = DateTimeOffset.UtcNow;
            var result = WarmHostLock.TryAcquire(path, StaleThreshold, () => now);

            Assert.Null(result);
            Assert.True(File.Exists(path)); // untouched -- not stale, so not reclaimed/deleted.
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Dispose_DeletesTheLockFile()
    {
        var path = TempLockPath();
        var handle = WarmHostLock.TryAcquire(path, StaleThreshold);
        Assert.NotNull(handle);
        Assert.True(File.Exists(path));

        handle!.Dispose();

        Assert.False(File.Exists(path));
    }

    [Fact]
    public void TryAcquire_AfterRelease_NewCallerCanAcquire()
    {
        var path = TempLockPath();
        try
        {
            var first = WarmHostLock.TryAcquire(path, StaleThreshold);
            Assert.NotNull(first);
            first!.Dispose();

            using var second = WarmHostLock.TryAcquire(path, StaleThreshold);
            Assert.NotNull(second);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
