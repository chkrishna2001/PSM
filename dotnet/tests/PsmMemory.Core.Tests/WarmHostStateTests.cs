using PsmMemory.Core.Runtime.WarmHost;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Covers <see cref="WarmHostState"/>'s tolerant read / atomic write behavior -- the daemon.json state
/// file whose <see cref="DaemonState.LastSeenAt"/> rewrite is the warm host's sliding-expiration
/// mechanism (see <see cref="WarmHostServer"/>).
/// </summary>
public class WarmHostStateTests
{
    private static string TempStatePath() => Path.Combine(Path.GetTempPath(), $"psm-warmhost-state-{Guid.NewGuid():N}.json");

    [Fact]
    public void WriteThenRead_RoundTripsAllFields()
    {
        var path = TempStatePath();
        try
        {
            var state = new DaemonState
            {
                Pid = 4242,
                Host = "127.0.0.1",
                Port = 51515,
                StartedAt = DateTimeOffset.Parse("2026-07-21T10:00:00Z"),
                LastSeenAt = DateTimeOffset.Parse("2026-07-21T10:05:30Z"),
            };

            WarmHostState.Write(path, state);
            var read = WarmHostState.Read(path);

            Assert.NotNull(read);
            Assert.Equal(state.Pid, read!.Pid);
            Assert.Equal(state.Host, read.Host);
            Assert.Equal(state.Port, read.Port);
            Assert.Equal(state.StartedAt, read.StartedAt);
            Assert.Equal(state.LastSeenAt, read.LastSeenAt);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Write_OverwritesPreviousContentCompletely_NotCorrupted()
    {
        var path = TempStatePath();
        try
        {
            WarmHostState.Write(path, new DaemonState { Pid = 1, Host = "127.0.0.1", Port = 1000, StartedAt = DateTimeOffset.UtcNow, LastSeenAt = DateTimeOffset.UtcNow });
            // Second write's serialized JSON is shorter-or-longer than the first in general -- writing
            // a state with meaningfully different field widths exercises that the old content doesn't
            // leak through if the atomic replace were somehow partial.
            var second = new DaemonState { Pid = 999999, Host = "127.0.0.1", Port = 65000, StartedAt = DateTimeOffset.UtcNow, LastSeenAt = DateTimeOffset.UtcNow.AddMinutes(5) };
            WarmHostState.Write(path, second);

            var read = WarmHostState.Read(path);
            Assert.NotNull(read);
            Assert.Equal(999999, read!.Pid);
            Assert.Equal(65000, read.Port);

            // No leftover .tmp file after a successful write.
            Assert.False(File.Exists(path + ".tmp"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Read_MissingFile_ReturnsNull()
    {
        var path = TempStatePath();
        Assert.False(File.Exists(path));
        Assert.Null(WarmHostState.Read(path));
    }

    [Fact]
    public void Read_CorruptJson_ReturnsNull()
    {
        var path = TempStatePath();
        try
        {
            File.WriteAllText(path, "{ this is not valid json ");
            Assert.Null(WarmHostState.Read(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Read_WrongShape_ReturnsNull()
    {
        var path = TempStatePath();
        try
        {
            // Valid JSON, but missing the required Host field and totally the wrong shape otherwise.
            File.WriteAllText(path, "{\"someOtherField\": 123}");
            Assert.Null(WarmHostState.Read(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
