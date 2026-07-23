using PsmMemory.Core.Runtime;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Covers the shared walk-up-from-a-starting-directory model dir resolver extracted from CLI's
/// Defaults.ResolveModelDir(). The MCP host previously duplicated only the "look right here"
/// half of this logic (no walk-up), which meant it failed to find the model whenever launched
/// from anywhere but the repo root -- these cases pin the walk-up behavior both hosts now share.
/// </summary>
public class ModelDirResolverTests : IDisposable
{
    private const string RelativeModelDir = "psm-model/prod-memory/gguf-runtime/v1";
    private readonly string _root;

    public ModelDirResolverTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "psm-modeldir-resolver-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root))
        {
            Directory.Delete(_root, recursive: true);
        }
    }

    [Fact]
    public void ResolveFromBaseDirectory_ModelDirPresentAtStart_ReturnsStartCandidate()
    {
        var startDir = Path.Combine(_root, "start");
        Directory.CreateDirectory(Path.Combine(startDir, RelativeModelDir));

        var resolved = ModelDirResolver.ResolveFromBaseDirectory(startDir, RelativeModelDir);

        Assert.Equal(Path.Combine(startDir, RelativeModelDir), resolved);
    }

    [Fact]
    public void ResolveFromBaseDirectory_ModelDirTwoLevelsUp_WalksUpAndFindsIt()
    {
        // root/ (has the model dir)
        //   root/a/
        //     root/a/b/  (start here)
        var levelA = Path.Combine(_root, "a");
        var levelB = Path.Combine(levelA, "b");
        Directory.CreateDirectory(levelB);
        Directory.CreateDirectory(Path.Combine(_root, RelativeModelDir));

        var resolved = ModelDirResolver.ResolveFromBaseDirectory(levelB, RelativeModelDir);

        Assert.Equal(Path.Combine(_root, RelativeModelDir), resolved);
    }

    [Fact]
    public void ResolveFromBaseDirectory_ModelDirAbsentEverywhere_ReturnsRelativePathUnchanged()
    {
        var startDir = Path.Combine(_root, "nowhere", "to", "be", "found");
        Directory.CreateDirectory(startDir);

        var resolved = ModelDirResolver.ResolveFromBaseDirectory(startDir, RelativeModelDir);

        Assert.Equal(RelativeModelDir, resolved);
    }
}
