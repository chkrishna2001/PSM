# PSM Memory TypeScript Stack

This tree separates reusable memory code from public commands and benchmarks:

- `src/psm-core`: npm package `@psm-memory/sdk`; SDK types, JSON parsing, action routing, SQLite memory store, ranking, prompt builders, and service orchestration.
- `src/psm-cli`: npm package `@psm-memory/cli`; thin `psm` command wrapper over the SDK.
- `src/psm-pi-plugin`: npm package `@psm-memory/pi-plugin`; tool registration helpers for agents/plugins.
- `benchmark/locomo`: benchmark-only LOCOMO ingestion and evidence retrieval tooling.

The public CLI intentionally exposes memory operations only:

```powershell
node src/psm-cli/dist/cli.js setup
node src/psm-cli/dist/cli.js remember "User prefers SQLite for local apps."
node src/psm-cli/dist/cli.js recall "What database does the user prefer?"
```

After publishing, the primary command is exposed as `psm-memory` by the `@psm-memory/cli` package. The shorter `psm` command is also available as an alias.

Recall returns readable text by default. Use `--json` when tooling needs structured output.

## Runtime

The npm CLI package downloads the default Q4_K_M GGUF model during package installation. The CLI uses this managed model automatically.

Set `PSM_MEMORY_SKIP_MODEL_DOWNLOAD=1` to skip the install-time download in CI or packaging environments. Users can run `psm-memory setup` later to download the model into the local PSM Memory cache.

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

## Grounding Invariant

Agent-injected memory context must be built from exact database rows. PSM may plan retrieval and rank candidates, but it must not generate free-form memory facts for injection. Every context item must carry the stored memory id/table and copied content from that row.
