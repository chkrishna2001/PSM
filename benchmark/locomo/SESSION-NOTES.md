# LOCOMO Benchmark Session Notes

Date: 2026-05-12

## Current Goal

Run LOCOMO ingestion through PSM using the quantized GGUF model and `node-llama-cpp`, then evaluate evidence retrieval from the SQLite memory DB.

## Model Artifacts

- F16 GGUF: `models\psm.gguf`
  - Converted from `psm_merged_fp16`
  - Size: `3,093,668,672` bytes
  - Final `llama-gguf-hash`: `xxh64 99e8af5191193fdf`
- Q4_K_M GGUF: `models\psm-q4_k_m.gguf`
  - Quantized from `models\psm.gguf`
  - Size: `986,047,808` bytes
  - Final `llama-gguf-hash`: `xxh64 b6b96f834dd9e73d`

## Conversion Commands Used

```powershell
psm\Scripts\python.exe C:\Users\chkri\source\repos\llama.cpp\convert_hf_to_gguf.py `
  psm_merged_fp16 `
  --outfile models\psm.gguf `
  --outtype f16
```

```powershell
C:\Users\chkri\source\repos\llama.cpp\build\bin\Release\llama-quantize.exe `
  models\psm.gguf `
  models\psm-q4_k_m.gguf `
  Q4_K_M
```

The converter needed `sentencepiece` installed into the existing `psm` venv.

## Benchmark Layout

- Dataset moved to: `benchmark\locomo\data\locomo10.json`
- LOCOMO source scripts:
  - `benchmark\locomo\src\ingest.ts`: llama-server HTTP ingestion
  - `benchmark\locomo\src\ingest-node.ts`: direct `node-llama-cpp` ingestion
  - `benchmark\locomo\src\evaluate.ts`: retrieval evaluation
- Runner scripts:
  - `benchmark\locomo\run-ingest.ps1`: starts/stops `llama-server`
  - `benchmark\locomo\run-ingest-node.ps1`: uses `node-llama-cpp` directly
  - `benchmark\locomo\run-evaluate.ps1`

## Runtime Findings

### llama.cpp server

The locally built `llama-server.exe` did not use GPU. With `-ngl 999`, logs showed:

```text
warning: no usable GPU found, --gpu-layers option will be ignored
```

It ran CPU-only. A 100-turn ingest through server took about `5m 53s`, estimating roughly `5-6h` for all 5,882 turns.

### node-llama-cpp

Installed package:

```powershell
npm install node-llama-cpp
```

Installed version: `3.18.1`.

`node-llama-cpp` successfully selected GPU via Vulkan:

```text
node-llama-cpp supported GPU backends: vulkan, false
node-llama-cpp selected backend: vulkan
node-llama-cpp model GPU layers: 29
```

So tomorrow use the Node ingestion path, not the `llama-server` path.

## Smoke Runs Completed

### Server path

- 10-turn smoke: `10 stored, 0 failed`
- 100-turn CPU/server run: `100 stored, 0 failed`
- 100-turn eval:
  - `hit_at_1 = 0.08121827411167512`
  - `hit_at_3 = 0.1319796954314721`

### node-llama-cpp path

- 2-turn GPU smoke: `2 stored, 0 failed`
- 100-turn GPU run with original long prompt: `100 stored, 0 failed`, about `8m 55s`
- 25-turn GPU run with compact prompt: `25 stored, 0 failed`, about `1m 17s`

The compact prompt is now in `benchmark\locomo\src\ingest-node.ts` and should be used for the full run.

## Commands For Tomorrow

Build first:

```powershell
npm run build
```

Optional smoke:

```powershell
powershell -ExecutionPolicy Bypass -File benchmark\locomo\run-ingest-node.ps1 `
  -Db benchmark\locomo\results\locomo-node-gpu-smoke.db `
  -Limit 25 `
  -BatchSize 5 `
  -Gpu auto `
  -GpuLayers auto
```

Full ingestion:

```powershell
Remove-Item -LiteralPath benchmark\locomo\results\locomo-psm-memory-node.db -Force -ErrorAction SilentlyContinue

powershell -ExecutionPolicy Bypass -File benchmark\locomo\run-ingest-node.ps1 `
  -Db benchmark\locomo\results\locomo-psm-memory-node.db `
  -BatchSize 100 `
  -Gpu auto `
  -GpuLayers auto
```

Evaluate after ingestion:

```powershell
powershell -ExecutionPolicy Bypass -File benchmark\locomo\run-evaluate.ps1 `
  -Db benchmark\locomo\results\locomo-psm-memory-node.db `
  -Out benchmark\locomo\results\locomo-node-results.json `
  -TopK 3
```

## Important Notes

- The interrupted full run was aborted almost immediately. If `locomo-psm-memory-node.db` exists, delete it before the full run unless intentionally resuming.
- `node:sqlite` prints an experimental warning on Node 22.17.0; this is expected and did not break tests or benchmark scripts.
- `node-llama-cpp` logs a tokenizer warning:

```text
control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
```

This warning appeared during successful smoke runs.

## Validation Commands

```powershell
npm test
```

Last known result: all tests passed.

