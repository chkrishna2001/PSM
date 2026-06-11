import { PsmModelRuntime } from "../../dist/src/psm-core/src/psm-model-runtime.js";
import { buildStoragePrompt } from "../../dist/src/psm-core/src/prompts.js";

process.env.PSM_FORCE_CPU = "1";
const runtime = new PsmModelRuntime({
  checkpoint: "psm-model/checkpoints/real-v3-50m-full-v2-step-048000.pt",
  python: ".venv/Scripts/python.exe",
  repoRoot: process.cwd(),
  device: "cpu"
});
const prompt = buildStoragePrompt(
  "Melanie said: Hey Caroline!",
  [],
  { source_kind: "locomo_turn", source_id: "test:1" },
  "Caroline said: Hey Mel!"
);
const start = Date.now();
const raw = await runtime.generateJson(prompt);
console.log("ms", Date.now() - start, raw.slice(0, 120));
