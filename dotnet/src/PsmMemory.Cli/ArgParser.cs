namespace PsmMemory.Cli;

/// <summary>
/// Thrown for any expected CLI usage problem (missing/invalid flag, bad value, unknown command).
/// Caught at the top level in Program.cs and reported as a clean one-line error instead of a stack trace.
/// </summary>
internal sealed class CliUsageException(string message) : Exception(message);

/// <summary>
/// Minimal hand-rolled flag parser for this CLI's flat "--flag value" style. A dedicated
/// System.CommandLine dependency was tried and deliberately dropped: the only version that
/// restores against net10.0 today is a brand-new 3.0.0-preview with an unfamiliar API, and this
/// CLI's surface (six commands, all with simple `--flag value` options, no nesting) doesn't need
/// its subcommand/binding machinery. This parser recognizes "--name value" pairs and bare
/// "--name" boolean flags (e.g. --help, --no-existing).
/// </summary>
internal sealed class ArgParser
{
    private readonly Dictionary<string, string> _values = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _flags = new(StringComparer.OrdinalIgnoreCase);

    public static ArgParser Parse(string[] args)
    {
        var parser = new ArgParser();
        for (var i = 0; i < args.Length; i++)
        {
            var token = args[i];
            if (!token.StartsWith("--", StringComparison.Ordinal))
                throw new CliUsageException($"Unexpected argument '{token}' (expected a --flag).");

            var name = token[2..];
            if (string.IsNullOrEmpty(name))
                throw new CliUsageException("Unexpected argument '--' with no flag name.");

            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                parser._values[name] = args[++i];
            }
            else
            {
                parser._flags.Add(name);
            }
        }
        return parser;
    }

    public bool HasFlag(string name) => _flags.Contains(name) || _values.ContainsKey(name);

    public string? GetString(string name) => _values.TryGetValue(name, out var v) ? v : null;

    public string GetString(string name, string defaultValue) => GetString(name) ?? defaultValue;

    public string GetRequiredString(string name)
    {
        var value = GetString(name);
        if (string.IsNullOrWhiteSpace(value))
            throw new CliUsageException($"Missing required flag --{name} <value>.");
        return value;
    }

    public int? GetInt(string name)
    {
        var raw = GetString(name);
        if (raw is null) return null;
        if (!int.TryParse(raw, out var value))
            throw new CliUsageException($"--{name} must be an integer, got '{raw}'.");
        return value;
    }

    public int GetInt(string name, int defaultValue) => GetInt(name) ?? defaultValue;
}
