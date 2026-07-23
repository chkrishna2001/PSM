namespace PsmMemory.Core.Runtime;

/// <summary>
/// Controls how <see cref="PsmDomainParser.Parse"/> handles an unrecognized (or missing) domain
/// string. There were previously three independent copies of this "coding|conversational" parsing
/// logic (CLI direct commands, CLI hook commands, MCP tools) with genuinely different behavior on
/// bad input -- this enum captures that real difference in one place instead of re-diverging it.
/// </summary>
public enum DomainParseMode
{
    /// <summary>Unrecognized (including null/empty) input throws <see cref="PsmDomainParseException"/>.</summary>
    Strict,

    /// <summary>Unrecognized (including null/empty) input silently falls back to <see cref="PsmDomain.Coding"/>,
    /// and this mode never throws. Used by hook commands, which must never hard-fail an agent's turn
    /// on a bad --domain value.</summary>
    LenientDefaultCoding,
}

/// <summary>Thrown by <see cref="PsmDomainParser.Parse"/> in <see cref="DomainParseMode.Strict"/> mode
/// for an unrecognized domain string. Callers translate this into whatever exception type/shape their
/// host (CLI vs MCP) expects to preserve their existing external behavior.</summary>
public sealed class PsmDomainParseException : Exception
{
    public PsmDomainParseException(string message) : base(message)
    {
    }
}

/// <summary>
/// Single shared implementation of "coding|conversational" string -&gt; <see cref="PsmDomain"/>
/// parsing, used by all three call sites (CLI direct commands, CLI hook commands, MCP tools) via
/// thin per-host wrappers that preserve each host's pre-existing external behavior on bad input.
/// </summary>
public static class PsmDomainParser
{
    public static PsmDomain Parse(string? raw, DomainParseMode mode)
    {
        var normalized = (raw ?? string.Empty).Trim().ToLowerInvariant();
        return normalized switch
        {
            "coding" => PsmDomain.Coding,
            "conversational" => PsmDomain.Conversational,
            _ when mode == DomainParseMode.LenientDefaultCoding => PsmDomain.Coding,
            _ => throw new PsmDomainParseException(
                $"domain must be one of coding|conversational, got '{raw}'."),
        };
    }
}
