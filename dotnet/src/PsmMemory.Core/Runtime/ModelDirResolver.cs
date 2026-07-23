namespace PsmMemory.Core.Runtime;

/// <summary>
/// Shared "walk up from a starting directory looking for a relative model directory" resolver.
/// Both hosts (CLI and MCP) need this, but from legitimately different starting points -- the CLI
/// walks up from <c>AppContext.BaseDirectory</c> (the running executable's directory), while the MCP
/// server walks up from <c>Directory.GetCurrentDirectory()</c> (its process's working directory,
/// since MCP clients launch it with a fixed command line and no way to pass flags). This class only
/// captures the shared walk-up algorithm; callers decide which directory to start from.
/// </summary>
public static class ModelDirResolver
{
    /// <summary>
    /// Walks up from <paramref name="startDirectory"/> looking for a directory whose contents include
    /// <paramref name="relativeModelDir"/>. Returns the first matching absolute path found, or
    /// <paramref name="relativeModelDir"/> unchanged if no ancestor directory contains it.
    /// </summary>
    public static string ResolveFromBaseDirectory(string startDirectory, string relativeModelDir)
    {
        var dir = new DirectoryInfo(startDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, relativeModelDir);
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return relativeModelDir;
    }
}
