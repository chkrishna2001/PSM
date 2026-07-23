using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using PsmMemory.Core;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;
using PsmMemory.Mcp;

var builder = Host.CreateApplicationBuilder(args);

// stdout is reserved for the MCP JSON-RPC protocol over the stdio transport; all diagnostic
// logging must go to stderr instead, or it will corrupt the protocol stream.
builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);

// --- Configuration (env vars only -- MCP servers are launched by a client with a fixed command
// line, not interactively, so there's nowhere to pass flags). ---
//
// PSM_DB_PATH: path to the SQLite memory database. Defaults to "user_memory.db" in the current
// working directory of the launched process (created if it doesn't exist).
//
// PSM_MODEL_DIR: path to the directory containing the exported PSM GGUF model, i.e. a directory
// with the base model GGUF, tokenizer files, and adapters for storage/retrieval_plan/consolidation.
// Defaults to "psm-model/prod-memory/gguf-runtime/v1", resolved by walking up from the server
// process's current working directory looking for that relative path (see ModelDirResolver) --
// this works whether the server is launched with the repo root as its working directory or from a
// subdirectory of the checkout. If your MCP client launches this tool from entirely outside the
// checkout, set PSM_MODEL_DIR explicitly to an absolute path.
var dbPath = Environment.GetEnvironmentVariable("PSM_DB_PATH") is { Length: > 0 } envDb
    ? envDb
    : "user_memory.db";

var modelDir = Environment.GetEnvironmentVariable("PSM_MODEL_DIR") is { Length: > 0 } envModel
    ? envModel
    : ModelDirResolver.ResolveFromBaseDirectory(Directory.GetCurrentDirectory(), LlamaSharpPsmRuntime.DefaultRelativeModelDirectory);

builder.Services.AddSingleton(_ =>
{
    var store = new MemoryStore(dbPath);
    store.InitializeSchema();
    return store;
});

// LlamaSharpPsmRuntime.CreateAsync downloads the GGUF model from HuggingFace first if modelDir
// doesn't already have it (first run), then loads the base model + all adapters -- slow (seconds to
// minutes on first download), so it must happen exactly once here at process startup, not per tool
// call. Awaited directly (top-level await) rather than via an AddSingleton factory, since DI's
// AddSingleton factories are synchronous; the already-constructed instance is then registered by value.
var runtime = await LlamaSharpPsmRuntime.CreateAsync(modelDir).ConfigureAwait(false);
builder.Services.AddSingleton<IPsmRuntime>(runtime);

// Same bootstrap pattern as the LLM runtime above: downloads the embedding GGUF from HuggingFace
// on first run if missing, then loads it once for the lifetime of this process. Domain-agnostic --
// backs vector recall for both PsmDomain.Coding and PsmDomain.Conversational tool calls.
var embeddingRuntime = await LlamaSharpEmbeddingRuntime.CreateAsync(modelDir).ConfigureAwait(false);
builder.Services.AddSingleton<IEmbeddingRuntime>(embeddingRuntime);

builder.Services.AddSingleton(sp => new PsmService(
    sp.GetRequiredService<MemoryStore>(),
    sp.GetRequiredService<IPsmRuntime>(),
    sp.GetRequiredService<IEmbeddingRuntime>()));

// RememberQueueDrainer is the single Core entry point for remember() -- every host (this MCP tool,
// the CLI, tests) goes through it, so there is exactly one processing path, not one path per host.
// This "foreground" instance shares the primary MemoryStore/PsmService above and is what
// PsmMemoryTools.Remember uses for its enqueue-only (fire-and-forget) call.
builder.Services.AddSingleton(sp => new RememberQueueDrainer(
    sp.GetRequiredService<MemoryStore>(),
    sp.GetRequiredService<PsmService>()));

// The background worker needs its OWN separate MemoryStore -- a second SqliteConnection to the same
// dbPath -- rather than sharing the foreground drainer/store above. MemoryStore has no internal
// locking, and the worker's poll loop is a second thread continuously touching the store; the
// MemoryStore constructor now sets WAL + busy_timeout specifically so this second connection can
// coexist safely with the foreground one. The LLM/embedding runtimes ARE shared (both already
// self-serialize via internal SemaphoreSlims), so reloading the GGUF model a second time would be
// pure waste. Constructed directly here (not resolved as the generic RememberQueueDrainer singleton
// above, which would give the worker the WRONG -- shared -- connection) and handed to
// RememberQueueWorker explicitly so DI never has to disambiguate between the two instances.
builder.Services.AddSingleton(sp =>
{
    var workerStore = new MemoryStore(dbPath);
    workerStore.InitializeSchema();
    var workerService = new PsmService(workerStore, sp.GetRequiredService<IPsmRuntime>(), sp.GetRequiredService<IEmbeddingRuntime>());
    var workerDrainer = new RememberQueueDrainer(workerStore, workerService);
    return new RememberQueueWorker(workerDrainer, sp.GetRequiredService<ILogger<RememberQueueWorker>>());
});
builder.Services.AddHostedService(sp => sp.GetRequiredService<RememberQueueWorker>());

builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

var app = builder.Build();
await app.RunAsync();
