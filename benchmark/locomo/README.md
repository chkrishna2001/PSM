# LOCOMO Benchmark

This folder contains the LOCOMO data, ingestion database, and benchmark outputs for the PSM memory path.

## Layout

- `data/locomo10.json`: LOCOMO dataset.
- `results/`: benchmark outputs.
- `src/ingest.ts`: batched ingestion through PSM.
- `src/evaluate.ts`: evidence retrieval evaluation over the SQLite memory DB.
- `psm-memory-locomo-colab.ipynb`: Colab notebook that installs the published npm packages and runs a LOCOMO smoke/full benchmark.
- `colab/locomo-benchmark.mjs`: Colab harness used by the notebook.

## Start llama.cpp Server

Keep the model loaded once and let ingestion call the server repeatedly:

```powershell
C:\Users\chkri\source\repos\llama.cpp\build\bin\Release\llama-server.exe `
  -m models\psm-q4_k_m.gguf `
  -c 4096 `
  -ngl 999 `
  --port 8080
```

If GPU layers are unsupported by the local build/runtime, remove `-ngl 999`.

## Ingest

Smoke test first:

```powershell
npm run build
benchmark\locomo\run-ingest.ps1 -Limit 25 -BatchSize 4
```

Full ingestion:

```powershell
benchmark\locomo\run-ingest.ps1 -BatchSize 4
```

The default DB is `benchmark\locomo\results\locomo-psm-memory.db`.

Use `-UseGpu` to pass `-ngl 999` to llama.cpp. The current observed run loaded on CPU; a CUDA-enabled llama.cpp build/runtime should show CUDA buffers in `benchmark\locomo\results\llama-server.err.log`.

## Ingest With node-llama-cpp

This path uses the Node runtime directly and asks `node-llama-cpp` to select GPU automatically:

```powershell
npm run build
benchmark\locomo\run-ingest-node.ps1 -Limit 25 -Gpu auto -GpuLayers auto
```

For a full run:

```powershell
benchmark\locomo\run-ingest-node.ps1 -Gpu auto -GpuLayers auto
```

If CUDA is available through `node-llama-cpp`, startup logs should show CUDA as the selected backend and a positive model GPU layer count.

## Evaluate

```powershell
benchmark\locomo\run-evaluate.ps1 -TopK 3
```

## Compare Against Published Memory Tool Results

For a true comparable score, run answer evaluation after ingest. This retrieves PSM memories, generates an answer for each LOCOMO question, then asks an LLM judge to score the generated answer against the gold answer. By default this uses OpenRouter with `nvidia/nemotron-3-super-120b-a12b:free`; set `OPENROUTER_API_KEY` before running.

```powershell
npm run build
$env:OPENROUTER_API_KEY = "sk-or-..."
node dist/benchmark/locomo/src/answer-evaluate.js --db benchmark/locomo/results/locomo-psm-memory.db --out benchmark/locomo/results/locomo-answer-results.json --top-k 50
```

Use `--answer-model`, `--judge-model`, and `--base-url` if you want a different OpenAI-compatible provider.

Then generate the comparison report from the answer result:

```powershell
node dist/benchmark/locomo/src/report.js --psm benchmark/locomo/results/locomo-answer-results.json --baselines benchmark/locomo/baselines/memory-tools.json --out benchmark/locomo/results/locomo-comparison.md
```

The older retrieval-only evaluation still reports evidence retrieval quality (`hit_at_1`, `hit_at_3`): it checks whether retrieved memories contain at least one gold LOCOMO evidence id. Use that as a diagnostic, not as the headline comparison score.

```powershell
npm run build
node dist/benchmark/locomo/src/report.js --psm benchmark/locomo/results/locomo-results.json --baselines benchmark/locomo/baselines/memory-tools.json --out benchmark/locomo/results/locomo-comparison.md
```

The Colab notebook writes `/content/locomo/results/locomo-answer-results.json`, `/content/locomo/results/locomo-comparison.md`, and copies both to `MyDrive/psm-memory-locomo/`.

Current timing on Q4_K_M CPU runtime: 100 LOCOMO turns took about 5 minutes 53 seconds, so the full 5,882-turn ingest is expected to take roughly 5-6 hours unless GPU offload is active.
