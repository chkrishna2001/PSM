import { HeuristicRuntime, MemoryStore, NodeLlamaRuntime, PsmService, memoryTables, type MemoryTable, type ModelRuntime } from "@psm-memory/sdk";
import { boolOption, intOption, parseArgs, required, stringOption } from "./args.js";

export async function run(argv: string[]): Promise<number> {
  const { command, options } = parseArgs(argv);
  const dbPath = stringOption(options, "db", "user_memory.db");
  const pretty = boolOption(options, "pretty");

  try {
    if (command === "help" || command === "--help" || command === "-h") {
      write(helpText());
      return 0;
    }

    const store = new MemoryStore(dbPath);
    try {
      if (command === "init") {
        store.initializeSchema();
        output({ db: dbPath, initialized: true }, pretty);
        return 0;
      }

      store.initializeSchema();
      const runtime = createRuntime(options);
      const service = new PsmService(store, runtime);

      if (command === "context") {
        output(await service.context({
          prompt: required(options, "prompt"),
          userId: stringOption(options, "user", "default-user"),
          topK: intOption(options, "top-k", 5)
        }), pretty);
        return 0;
      }

      if (command === "remember") {
        output(await service.remember({
          llmResponse: required(options, "llm-response"),
          userId: stringOption(options, "user", "default-user")
        }), pretty);
        return 0;
      }

      if (command === "recall") {
        output(await service.recall({
          question: required(options, "question"),
          userId: stringOption(options, "user", "default-user"),
          topK: intOption(options, "top-k", 5)
        }), pretty);
        return 0;
      }

      if (command === "show") {
        const table = stringOption(options, "table", "episodic") as MemoryTable;
        if (!memoryTables.includes(table)) throw new Error(`Unsupported table: ${table}`);
        output(store.selectTable(table, intOption(options, "limit", 20)), pretty);
        return 0;
      }

      if (command === "conflicts") {
        output(store.selectConflicts(stringOption(options, "status", "unresolved"), intOption(options, "limit", 20)), pretty);
        return 0;
      }

      throw new Error(`Unknown command: ${command}`);
    } finally {
      store.close();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    return 1;
  }
}

function createRuntime(options: Record<string, string | boolean>): ModelRuntime {
  const modelPath = stringOption(options, "model", "");
  if (modelPath) {
    return new NodeLlamaRuntime({
      modelPath,
      contextSize: intOption(options, "context-size", 4096),
      gpu: stringOption(options, "gpu", "auto") as "auto",
      gpuLayers: stringOption(options, "gpu-layers", "auto") as "auto"
    });
  }
  return new HeuristicRuntime();
}

function output(value: unknown, pretty: boolean): void {
  process.stdout.write(`${JSON.stringify(value, null, pretty ? 2 : 0)}\n`);
}

function write(value: string): void {
  process.stdout.write(value);
}

function helpText(): string {
  return `PSM Memory commands:
  init --db <path>
  context --prompt <text> --user <id> --db <path> [--top-k n] [--model psm.gguf] [--pretty]
  remember --llm-response <text> --user <id> --db <path> [--model psm.gguf] [--pretty]
  recall --question <text> --user <id> --db <path> [--top-k n] [--model psm.gguf] [--pretty]
  show --table episodic|semantic|archival|conflicts|decisions|decay_schedule --db <path> [--limit n] [--pretty]
  conflicts --status unresolved|resolved|dismissed --db <path> [--limit n] [--pretty]

JSON output is the default. Omit --model to use deterministic fallback routing for local tests.
`;
}
