using System.Text.Json;

namespace PsmMemory.Core.Runtime.WarmHost;

/// <summary>
/// Wire request DTO for the warm host's loopback HTTP protocol (see <see cref="WarmHostServer"/> /
/// <see cref="WarmHostClient"/>). Mirrors src/psm-cli/src/daemon.ts's <c>POST /v1 {operation,
/// payload}</c> dispatch idiom and the general flavor of PsmMemory.Cli.Commands's NDJSON `serve`
/// command's cmd-dispatch -- one flat request type, one flat response type, four supported
/// operations. Unlike the TS original, there is no silent fallback-to-field-name behavior for missing
/// required fields here: callers populate the fields their chosen operation needs, and the server
/// returns an explicit <c>Error</c> (e.g. "missing_prompt") when they don't.
/// </summary>
public sealed class WarmHostRequest
{
    /// <summary>"storage-decision" | "recall-plan" | "consolidation-decision" | "embed".</summary>
    public required string Operation { get; set; }

    /// <summary>Used by the three generate operations.</summary>
    public string? Prompt { get; set; }

    /// <summary>"coding" | "conversational" -- serialized as <see cref="PsmDomain"/>'s string name.</summary>
    public string Domain { get; set; } = "coding";

    /// <summary>Used by "embed".</summary>
    public string? Text { get; set; }
}

public sealed class WarmHostResponse
{
    public bool Ok { get; set; }

    /// <summary>The generated string, for the three generate operations.</summary>
    public string? Result { get; set; }

    /// <summary>For "embed".</summary>
    public float[]? Embedding { get; set; }

    public string? Error { get; set; }
}

/// <summary>Shared <see cref="JsonSerializerOptions"/> for the warm host wire protocol -- camelCase,
/// case-insensitive on read -- used identically by <see cref="WarmHostServer"/> and
/// <see cref="WarmHostClient"/> (and <see cref="WarmHostState"/>) so every reader/writer of this
/// protocol serializes symmetrically.</summary>
internal static class WarmHostJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);
}
