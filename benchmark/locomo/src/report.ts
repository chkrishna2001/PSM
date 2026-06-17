import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

interface ReportOptions {
  psm: string;
  baselines: string;
  out: string;
}

interface PsmResults {
  summary?: {
    questions?: number;
    hit_at_1?: number;
    [key: string]: unknown;
  };
}

interface Baseline {
  system: string;
  score: number;
  metric: string;
  setup: string;
  source: string;
}

export function main(argv: string[]): number {
  const options = parseReportOptions(argv);
  const psm = JSON.parse(readFileSync(options.psm, "utf8")) as PsmResults;
  const baselines = JSON.parse(readFileSync(options.baselines, "utf8")) as Baseline[];
  const markdown = renderReport(psm, baselines);

  mkdirSync(dirname(options.out), { recursive: true });
  writeFileSync(options.out, markdown, "utf8");
  process.stdout.write(`${markdown}\nWrote ${options.out}\n`);
  return 0;
}

function renderReport(psm: PsmResults, baselines: Baseline[]): string {
  const summary = psm.summary ?? {};
  const answerAccuracy = typeof summary.answer_accuracy === "number" ? summary.answer_accuracy : undefined;
  const topKEntry = Object.entries(summary).find(([key]) => /^(?:evidence_)?hit_at_(?:\d+|k)$/.test(key) && !key.endsWith("hit_at_1"));
  const topKLabel = topKEntry?.[0].replace(/^(?:evidence_)?hit_at_/, "Hit@") ?? "Hit@K";
  const topKValue = typeof topKEntry?.[1] === "number" ? topKEntry[1] : undefined;
  const questions = typeof summary.questions === "number" ? String(summary.questions) : "";
  const hitAt1 = typeof summary.hit_at_1 === "number" ? summary.hit_at_1 : (typeof summary.evidence_hit_at_1 === "number" ? summary.evidence_hit_at_1 : undefined);
  const sortedBaselines = [...baselines].sort((a, b) => b.score - a.score);

  return [
    "# LOCOMO Memory Benchmark Comparison",
    "",
    answerAccuracy == null
      ? "This report places the local PSM retrieval run next to published memory-tool results. The PSM run is currently an evidence-retrieval benchmark: it measures whether a gold LOCOMO evidence `dia_id` appears in retrieved memories. Most public tool results below are answer-generation benchmarks scored by an LLM judge, so compare directionally and keep the metric column visible."
      : "This report places the local PSM answer-generation run next to published memory-tool results. PSM answer accuracy is generated from retrieved PSM memories and scored by an LLM judge, matching the broad LOCOMO scoring style used by public memory-tool reports. Exact numbers still depend on answer model, judge model, top-k, and prompt choices.",
    "",
    "## PSM Memory",
    "",
    "| System | Metric | Score | Questions | Notes |",
    "| --- | --- | ---: | ---: | --- |",
    ...(answerAccuracy == null ? [] : [
      `| PSM Memory | LoCoMo LLM-as-judge answer accuracy | ${formatPercent(answerAccuracy)} | ${questions} | Answer model: ${escapeCell(String(summary.answer_model ?? ""))}; judge model: ${escapeCell(String(summary.judge_model ?? ""))}; top-k: ${escapeCell(String(summary.top_k ?? ""))}. |`
    ]),
    `| PSM Memory | Evidence Hit@1 | ${formatPercent(hitAt1)} | ${questions} | Retrieved memory contains at least one gold evidence id in the first result. |`,
    `| PSM Memory | Evidence ${topKLabel} | ${formatPercent(topKValue)} | ${questions} | Retrieved memory contains at least one gold evidence id in the top-k set. |`,
    "",
    "## Published Memory Tool Results",
    "",
    "| System | Score | Metric | Setup | Source |",
    "| --- | ---: | --- | --- | --- |",
    ...sortedBaselines.map((baseline) => `| ${escapeCell(baseline.system)} | ${baseline.score.toFixed(2)}% | ${escapeCell(baseline.metric)} | ${escapeCell(baseline.setup)} | ${baseline.source} |`),
    "",
    "## Interpretation",
    "",
    ...(answerAccuracy == null ? [
      "- PSM numbers are not yet directly comparable to Mem0/Zep/Letta-style LoCoMo scores because they stop at retrieval and do not generate or judge final answers.",
      "- To make PSM fully comparable, add an answerer step over retrieved memories and score answers with the same judge/model/settings used by the target baseline.",
      "- Until then, PSM Evidence Hit@K is useful for diagnosing memory retrieval quality and estimating whether answer accuracy has enough evidence to improve."
    ] : [
      "- Use PSM answer accuracy as the comparable headline score.",
      "- Keep answer model, judge model, top-k, and prompt text attached to the result because LOCOMO scores are sensitive to these settings.",
      "- Evidence Hit@K remains useful as a retrieval diagnostic, but the leaderboard comparison should use answer accuracy."
    ])
  ].join("\n");
}

function formatPercent(value: number | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "";
}

function escapeCell(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\n/g, " ");
}

function parseReportOptions(argv: string[]): ReportOptions {
  const options: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      options[key] = next;
      i++;
    } else {
      options[key] = true;
    }
  }

  return {
    psm: stringOption(options, "psm", "benchmark/locomo/results/locomo-results.json"),
    baselines: stringOption(options, "baselines", "benchmark/locomo/baselines/memory-tools.json"),
    out: stringOption(options, "out", "benchmark/locomo/results/locomo-comparison.md")
  };
}

function stringOption(options: Record<string, string | boolean>, key: string, fallback: string): string {
  const value = options[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

if (process.argv[1]?.endsWith("report.js")) {
  process.exitCode = main(process.argv.slice(2));
}
