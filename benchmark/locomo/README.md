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

Current timing on Q4_K_M CPU runtime: 100 LOCOMO turns took about 5 minutes 53 seconds, so the full 5,882-turn ingest is expected to take roughly 5-6 hours unless GPU offload is active.
