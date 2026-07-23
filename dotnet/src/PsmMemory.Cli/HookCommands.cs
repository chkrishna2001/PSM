using System.Text.Json;
using PsmMemory.Core;
using PsmMemory.Core.Models;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;

namespace PsmMemory.Cli;

/// <summary>
/// `psm-memory hook &lt;mode&gt;` -- the agent-harness integration points ported from
/// src/psm-cli/src/index.ts's runHookRecall/runHookRemember/runHookSession. Every mode: (1) reads a
/// JSON blob from stdin (CLI flags only for options -- see HookIo.ParseHookInput), (2) never lets an
/// internal failure propagate as a nonzero exit (agent harnesses invoke these on every prompt/turn;
/// a hard failure here must not break the calling agent's turn), and (3) appends one line to the
/// hook audit log (see HookIo.AppendAuditLog) recording what happened either way.
/// </summary>
internal static class HookCommands
{
    public static async Task<int> RunAsync(string[] args)
    {
        string? mode = null;
        var remainder = args;
        if (args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal))
        {
            mode = args[0];
            remainder = args[1..];
        }

        var parsed = ArgParser.Parse(remainder);
        mode ??= parsed.GetString("mode");

        if (parsed.HasFlag("help"))
        {
            Console.WriteLine(HelpText.For(HelpText.Hook));
            return 0;
        }

        return mode switch
        {
            "recall" or "context" => await RunRecallAsync(parsed).ConfigureAwait(false),
            "remember" => await RunRememberAsync(parsed).ConfigureAwait(false),
            "session-start" => await RunSessionAsync(parsed, "session-start").ConfigureAwait(false),
            "session-end" => await RunSessionAsync(parsed, "session-end").ConfigureAwait(false),
            _ => throw new CliUsageException("Usage: psm-memory hook recall|remember|session-start|session-end"),
        };
    }

    private static Task<string> ReadStdinAsync() => Console.In.ReadToEndAsync();

    /// <summary>Hooks never hard-fail the caller's agent harness on a bad/unknown --domain value --
    /// fall back to Coding rather than throwing (unlike the direct recall/context/remember commands,
    /// which do validate --domain strictly).</summary>
    private static PsmDomain ParseHookDomain(ArgParser parsed) =>
        PsmDomainParser.Parse(parsed.GetString("domain"), DomainParseMode.LenientDefaultCoding);

    /// <summary>Hook commands are installed without a --user flag (see InstallAgentCommand's hook
    /// command strings), so unlike the direct commands, --user is optional here and falls back to the
    /// OS user name -- mirrors TS's defaultPsmUserId() (userInfo().username, else "local-user").</summary>
    private static string ResolveHookUserId(ArgParser parsed)
    {
        var explicitUser = parsed.GetString("user");
        if (!string.IsNullOrWhiteSpace(explicitUser)) return explicitUser;
        try
        {
            var envUser = Environment.UserName;
            return string.IsNullOrWhiteSpace(envUser) ? "local-user" : envUser;
        }
        catch
        {
            return "local-user";
        }
    }

    private static void WriteGeminiSuppressOutput() => Console.WriteLine(JsonSerializer.Serialize(new { suppressOutput = true }));

    public static async Task<int> RunRecallAsync(ArgParser parsed)
    {
        var agent = parsed.GetString("agent");
        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var ok = true;
        string? error = null;
        try
        {
            var stdin = await ReadStdinAsync().ConfigureAwait(false);
            var input = HookIo.ParseHookInput(stdin);
            var prompt = HookIo.FirstNonEmptyString(input, "prompt", "user_prompt", "message", "input");

            var renderedText = "";
            if (!string.IsNullOrWhiteSpace(prompt))
            {
                var userId = ResolveHookUserId(parsed);
                var topK = parsed.GetInt("top-k");
                var modelDir = parsed.GetString("model-dir", Defaults.ResolveModelDir());
                Defaults.EnsureModelDir(modelDir);
                var domain = ParseHookDomain(parsed);

                using var store = new MemoryStore(dbPath);
                store.InitializeSchema();
                await using var acquired = await Commands.AcquireRuntimeAsync(parsed, modelDir, dbPath).ConfigureAwait(false);
                var service = new PsmService(store, acquired.Runtime, acquired.EmbeddingRuntime);

                var result = await service.ContextAsync(new ContextRequest
                {
                    Prompt = prompt,
                    UserId = userId,
                    TopK = topK,
                    Domain = domain,
                }).ConfigureAwait(false);

                // Forward-compat: prefer a future RecallResult.AgentContext over this fallback once
                // Phase 3 exists -- see HookContextRenderer's doc comment.
                renderedText = HookContextRenderer.Render(result);
            }

            if (string.Equals(agent, "gemini", StringComparison.OrdinalIgnoreCase))
            {
                // Gemini CLI's BeforeAgent hook contract: always emit this exact JSON shape, even
                // when there's nothing to add.
                object payload = !string.IsNullOrEmpty(renderedText)
                    ? new
                    {
                        hookSpecificOutput = new { hookEventName = "BeforeAgent", additionalContext = renderedText },
                        suppressOutput = true,
                    }
                    : new { suppressOutput = true };
                Console.WriteLine(JsonSerializer.Serialize(payload));
            }
            else if (!string.IsNullOrEmpty(renderedText))
            {
                // Claude Code / Codex convention: raw stdout becomes additional context for the turn.
                Console.WriteLine(renderedText);
            }
        }
        catch (Exception ex)
        {
            ok = false;
            error = ex.Message;
        }
        finally
        {
            HookIo.AppendAuditLog(dbPath, "recall", agent, ok, error, Environment.GetEnvironmentVariable("PSM_MEMORY_HOOK_LOG"));
        }
        return 0;
    }

    public static async Task<int> RunRememberAsync(ArgParser parsed)
    {
        var agent = parsed.GetString("agent");
        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var ok = true;
        string? error = null;
        try
        {
            var stdin = await ReadStdinAsync().ConfigureAwait(false);
            var input = HookIo.ParseHookInput(stdin);

            var directResponse = HookIo.FirstNonEmptyString(input,
                "prompt_response", "last_assistant_message", "response", "assistant_response", "output", "text");
            var transcriptPath = HookIo.FirstNonEmptyString(input, "transcript_path", "transcriptPath");

            string? response = directResponse;
            var sourceKind = "hook_input";
            string? sourceId = null;

            if (response is null && !string.IsNullOrEmpty(transcriptPath))
            {
                var transcriptResponse = HookIo.ExtractLastAssistantMessageFromTranscriptFile(transcriptPath);
                if (transcriptResponse is not null)
                {
                    response = transcriptResponse;
                    sourceKind = "transcript";
                    sourceId = transcriptPath;
                }
            }

            if (response is null)
            {
                var latestPath = HookIo.LatestCodexSessionPath();
                var latestResponse = HookIo.ExtractLastAssistantMessageFromTranscriptFile(latestPath);
                if (latestResponse is not null)
                {
                    response = latestResponse;
                    sourceKind = "latest_codex_session";
                    sourceId = latestPath;
                }
            }

            if (response is not null)
            {
                var userId = ResolveHookUserId(parsed);
                var domain = ParseHookDomain(parsed);

                using var store = new MemoryStore(dbPath);
                store.InitializeSchema();
                // Fire-and-forget by design (matches TS's dispatchDaemonRemember for this hook) --
                // enqueue only, never RememberAsync/RememberAndWaitAsync directly.
                RememberQueueDrainer.Enqueue(store, new RememberRequest
                {
                    LlmResponse = response,
                    UserId = userId,
                    Domain = domain,
                    Source = new MemorySourceMetadata
                    {
                        SourceKind = sourceKind,
                        SourceId = sourceId,
                        SourceTimestamp = DateTime.UtcNow.ToString("o"),
                        SourceLabel = $"agent:{sourceKind}",
                    },
                });
            }
            // else: nothing resolved -- silent no-op, matches TS's "missing_response" skip (no error).

            if (string.Equals(agent, "gemini", StringComparison.OrdinalIgnoreCase)) WriteGeminiSuppressOutput();
        }
        catch (Exception ex)
        {
            ok = false;
            error = ex.Message;
        }
        finally
        {
            HookIo.AppendAuditLog(dbPath, "remember", agent, ok, error, Environment.GetEnvironmentVariable("PSM_MEMORY_HOOK_LOG"));
        }
        return 0;
    }

    public static async Task<int> RunSessionAsync(ArgParser parsed, string mode)
    {
        var agent = parsed.GetString("agent");
        var dbPath = parsed.GetString("db", Defaults.DbPath);
        var ok = true;
        string? error = null;
        try
        {
            var stdin = await ReadStdinAsync().ConfigureAwait(false);
            var input = HookIo.ParseHookInput(stdin);
            var transcriptPath = HookIo.FirstNonEmptyString(input, "transcript_path", "transcriptPath");

            var summary = BuildSessionSummary(mode, agent, transcriptPath);

            var userId = ResolveHookUserId(parsed);
            var domain = ParseHookDomain(parsed);

            using var store = new MemoryStore(dbPath);
            store.InitializeSchema();
            RememberQueueDrainer.Enqueue(store, new RememberRequest
            {
                LlmResponse = summary,
                UserId = userId,
                Domain = domain,
                Source = new MemorySourceMetadata
                {
                    SourceKind = mode == "session-start" ? "session_start" : "session_end",
                    SourceId = transcriptPath,
                    SourceTimestamp = DateTime.UtcNow.ToString("o"),
                    SourceLabel = $"agent {mode}",
                },
            });

            if (string.Equals(agent, "gemini", StringComparison.OrdinalIgnoreCase)) WriteGeminiSuppressOutput();
        }
        catch (Exception ex)
        {
            ok = false;
            error = ex.Message;
        }
        finally
        {
            HookIo.AppendAuditLog(dbPath, mode, agent, ok, error, Environment.GetEnvironmentVariable("PSM_MEMORY_HOOK_LOG"));
        }
        return 0;
    }

    /// <summary>Ported from TS's buildSessionSummary(): header line + agent + best-effort
    /// package.json name + best-effort git repo/branch/dirty-file info (swallowed entirely if git or
    /// the repo is unavailable) + transcript path if present.</summary>
    internal static string BuildSessionSummary(string mode, string? agent, string? transcriptPath)
    {
        var cwd = Directory.GetCurrentDirectory();
        var repo = GitOutput(cwd, "rev-parse --show-toplevel");
        var gitCwd = !string.IsNullOrEmpty(repo) ? repo : cwd;
        var branch = GitOutput(gitCwd, "branch --show-current");
        var dirty = GitOutput(gitCwd, "status --short")
            ?.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries)
            .Take(12)
            .ToList() ?? new List<string>();
        var packageName = PackageNameFor(gitCwd);

        var header = mode == "session-start" ? "Developer session started." : "Developer session ended.";
        var lines = new List<string> { header };
        if (!string.IsNullOrEmpty(agent)) lines.Add($"Agent: {agent}.");
        if (!string.IsNullOrEmpty(packageName)) lines.Add($"Project: {packageName}.");
        lines.Add(!string.IsNullOrEmpty(repo) ? $"Repo: {repo}." : $"CWD: {cwd}.");
        if (!string.IsNullOrEmpty(branch)) lines.Add($"Branch: {branch}.");
        lines.Add(dirty.Count > 0 ? $"Changed files: {string.Join("; ", dirty)}." : "Changed files: none detected.");
        if (!string.IsNullOrEmpty(transcriptPath)) lines.Add($"Transcript: {transcriptPath}.");
        return string.Join("\n", lines);
    }

    /// <summary>Best-effort: reads &lt;cwd&gt;/package.json's "name" field with plain JSON parsing.
    /// Missing file, unreadable file, or a missing/non-string "name" all resolve to null.</summary>
    internal static string? PackageNameFor(string cwd)
    {
        var path = Path.Combine(cwd, "package.json");
        if (!File.Exists(path)) return null;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.ValueKind == JsonValueKind.Object
                && doc.RootElement.TryGetProperty("name", out var nameEl)
                && nameEl.ValueKind == JsonValueKind.String)
            {
                return nameEl.GetString();
            }
        }
        catch
        {
            // best-effort only
        }
        return null;
    }

    private static string? GitOutput(string cwd, string arguments)
    {
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo("git", arguments)
            {
                WorkingDirectory = cwd,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var process = System.Diagnostics.Process.Start(psi);
            if (process is null) return null;
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);
            return process.ExitCode == 0 ? (string.IsNullOrWhiteSpace(output) ? null : output.Trim()) : null;
        }
        catch
        {
            return null;
        }
    }
}
