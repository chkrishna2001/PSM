# PSM Memory TypeScript Stack

This tree separates reusable memory code from public commands and benchmarks:

- `src/psm-core`: npm package `@psm-memory/sdk`; SDK types, JSON parsing, action routing, SQLite memory store, ranking, prompt builders, and service orchestration.
- `src/psm-cli`: npm package `@psm-memory/cli`; thin `psm` command wrapper over the SDK.
- `src/psm-pi-plugin`: npm package `@psm-memory/pi-plugin`; tool registration helpers for agents/plugins.
- `benchmark/locomo`: benchmark-only LOCOMO ingestion and evidence retrieval tooling.

The public CLI intentionally exposes memory operations only:

```powershell
node src/psm-cli/dist/cli.js init --db user_memory.db
node src/psm-cli/dist/cli.js context --prompt "What should I know?" --user demo --db user_memory.db --pretty
node src/psm-cli/dist/cli.js remember --llm-response "User prefers SQLite for local apps." --user demo --db user_memory.db --pretty
node src/psm-cli/dist/cli.js recall --question "What database does the user prefer?" --user demo --db user_memory.db --pretty
```

After publishing, the same command is exposed as `psm` by the `@psm-memory/cli` package.

JSON is the default output format. `--pretty` only changes formatting.

## Runtime

Without `--model`, the CLI uses a deterministic fallback runtime for local tests. Passing `--model psm.gguf` selects `NodeLlamaRuntime`, which is the intended GGUF integration point for `node-llama-cpp`. The source keeps `node-llama-cpp` optional so the SDK and CLI can build in environments where native dependencies are not installed.

The merged local model was converted with llama.cpp:

```powershell
psm\Scripts\python.exe C:\Users\chkri\source\repos\llama.cpp\convert_hf_to_gguf.py `
  psm_merged_fp16 `
  --outfile models\psm.gguf `
  --outtype f16
```

Generated artifact:

- `models\psm.gguf`
- Format: GGUF F16
- Size: about 3.09 GB
- llama.cpp tensor hash summary: final file `xxh64 99e8af5191193fdf`

Runtime quantized artifact:

```powershell
C:\Users\chkri\source\repos\llama.cpp\build\bin\Release\llama-quantize.exe `
  models\psm.gguf `
  models\psm-q4_k_m.gguf `
  Q4_K_M
```

- `models\psm-q4_k_m.gguf`
- Format: GGUF Q4_K_M
- Size: about 986 MB
- llama.cpp tensor hash summary: final file `xxh64 b6b96f834dd9e73d`

## SQLite Schema

The TypeScript store mirrors the existing memory tables:

- `episodic`
- `semantic`
- `archival`
- `conflicts`
- `decay_schedule`
- `decisions`

## Benchmark

LOCOMO stays outside the public CLI. The active benchmark folder is `benchmark\locomo`; the dataset is stored at `benchmark\locomo\data\locomo10.json`.

```powershell
npm run build
benchmark\locomo\run-ingest.ps1 -Limit 25 -BatchSize 4
benchmark\locomo\run-evaluate.ps1 -TopK 3
```

The benchmark uses SDK storage and ranking helpers so recall behavior is aligned with the CLI path.
