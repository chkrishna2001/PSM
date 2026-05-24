#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const manifestPath = process.argv[2] ?? "nano-psm/data-pipeline/sources/psm-source-manifest.json";
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const sources = Array.isArray(manifest.sources) ? manifest.sources : [];

for (const source of sources) {
  if (!source.expected_raw_dir) continue;
  mkdirSync(source.expected_raw_dir, { recursive: true });
  writeFileSync(join(source.expected_raw_dir, "README.md"), renderSourceReadme(source), "utf8");
}

const adapterPlanPath = "nano-psm/data-pipeline/sources/adapter-plan.md";
mkdirSync(dirname(adapterPlanPath), { recursive: true });
writeFileSync(adapterPlanPath, renderAdapterPlan(sources), "utf8");

console.log(JSON.stringify({
  manifest: manifestPath,
  prepared_sources: sources.length,
  adapter_plan: adapterPlanPath
}, null, 2));

function renderSourceReadme(source) {
  const refs = Array.isArray(source.public_refs) && source.public_refs.length > 0
    ? source.public_refs.map((ref) => `- ${ref}`).join("\n")
    : "- Local project dataset or manually downloaded benchmark.";
  const focus = Array.isArray(source.training_focus)
    ? source.training_focus.map((item) => `- ${item}`).join("\n")
    : "- Memory training source.";
  const commands = hfCommands(source);
  return `# ${source.name}

Role: ${source.role}

Adapter: \`${source.adapter}\`

Expected raw files live in this directory. Keep large downloaded datasets out of git.

Public references:

${refs}

Training focus:

${focus}

Suggested acquisition:

\`\`\`powershell
${commands.join("\n")}
\`\`\`
`;
}

function hfCommands(source) {
  const hfRef = (source.public_refs ?? []).find((ref) => ref.includes("huggingface.co/datasets/"));
  if (!hfRef) return ["# Download this source manually according to the paper/dataset instructions."];
  const repo = hfRef.split("huggingface.co/datasets/")[1]?.replace(/\/$/, "");
  if (!repo) return ["# Download this source manually according to the dataset instructions."];
  return [
    "# Requires: pip install -U huggingface_hub",
    `hf download ${repo} --repo-type dataset --local-dir ${source.expected_raw_dir}`
  ];
}

function renderAdapterPlan(sources) {
  const rows = sources.map((source) => `| ${source.name} | ${source.role} | ${source.adapter} | ${(source.target_operations ?? []).join(", ")} |`).join("\n");
  return `# PSM External Source Adapter Plan

| Source | Role | Adapter | Target Operations |
|---|---|---|---|
${rows}

Adapter contract:

1. Read raw source records from \`expected_raw_dir\`.
2. Normalize them into \`{ instruction, input, output }\` rows.
3. Preserve source ids and evidence ids in \`input\`.
4. Emit canonical PSM output keys only.
5. Add \`indexables\` for every stored memory.
6. Use \`recall_context\` rows for retrieval/reconstruction tasks.
7. Validate with \`validate-examples.mjs\`.

Do not train directly on source QA answers as memory ingestion labels. Convert them into recall-context selection labels so PSM learns what memory rows and indexable keys to activate.
`;
}
