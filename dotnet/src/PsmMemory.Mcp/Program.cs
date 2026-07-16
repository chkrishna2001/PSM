using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using PsmMemory.Core;
using PsmMemory.Core.Runtime;
using PsmMemory.Core.Store;

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
// PSM_MODEL_DIR: path to the directory containing the exported PSM ONNX model, i.e. a directory
// with model.onnx, tokenizer files, and an adapters/ subfolder holding storage.onnx_adapter,
// retrieval_plan.onnx_adapter, and consolidation.onnx_adapter. Defaults to
// "psm-model/prod-memory/onnx-runtime/v1" resolved against the current working directory --
// this only resolves correctly if the server is launched with the repo root as its working
// directory. If your MCP client launches this tool from elsewhere, set PSM_MODEL_DIR explicitly
// to an absolute path.
var dbPath = Environment.GetEnvironmentVariable("PSM_DB_PATH") is { Length: > 0 } envDb
    ? envDb
    : "user_memory.db";

var modelDir = Environment.GetEnvironmentVariable("PSM_MODEL_DIR") is { Length: > 0 } envModel
    ? envModel
    : Path.Combine(Directory.GetCurrentDirectory(), OnnxPsmRuntime.DefaultRelativeModelDirectory);

builder.Services.AddSingleton(_ =>
{
    var store = new MemoryStore(dbPath);
    store.InitializeSchema();
    return store;
});

// OnnxPsmRuntime.CreateAsync downloads the ONNX model from HuggingFace first if modelDir doesn't
// already have it (first run), then loads the base model + all adapters -- slow (seconds to minutes
// on first download), so it must happen exactly once here at process startup, not per tool call.
// Awaited directly (top-level await) rather than via an AddSingleton factory, since DI's AddSingleton
// factories are synchronous; the already-constructed instance is then registered by value.
var runtime = await OnnxPsmRuntime.CreateAsync(modelDir).ConfigureAwait(false);
builder.Services.AddSingleton<IPsmRuntime>(runtime);

builder.Services.AddSingleton(sp => new PsmService(sp.GetRequiredService<MemoryStore>(), sp.GetRequiredService<IPsmRuntime>()));

builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

var app = builder.Build();
await app.RunAsync();
