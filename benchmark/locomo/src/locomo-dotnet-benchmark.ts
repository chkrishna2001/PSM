/**
 * Full LoCoMo answer-accuracy benchmark run against the ACTUAL shipped PsmMemory.Cli .NET product
 * (not the parallel TS-native PsmService/MemoryStore used by ingest-psm-model.ts/answer-evaluate.ts).
 *
 * Ingests every turn of the held-out conversations (conv-30/41/42/43/44 -- never used in training
 * of any of the 3 conversational adapters, see build_storage_locomo_rows.py /
 * build_recall_locomo_rows.py's TRAIN/EVAL split) via `dotnet PsmMemory.Cli.dll remember --domain
 * conversational`, then answers every QA pair via `... recall --domain conversational` + an external
 * LLM to synthesize a final answer from the retrieved memories + judge it against the gold answer --
 * the same answer/judge/report methodology as answer-evaluate.ts, reused here as one self-contained
 * script since that file's helpers are module-local (not exported) and tag-based evidence linkage
 * doesn't directly apply to the .NET side's dedicated sourceId field.
 *
 * Talks to ONE long-lived `dotnet PsmMemory.Cli.dll serve` process over NDJSON stdin/stdout so the
 * ~2GB model+adapters load exactly once for the whole run, not once per remember/recall call.
 */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { loadSamples, filterSamples, parseSampleIds, flattenTurns, locomoSourceTimestamp } from "./common.js";
import { chatCompletion, resolveChatProvider, defaultAnswerModel, defaultJudgeModel, type ChatProviderConfig } from "./cloudflare-ai.js";
import type { LocomoSample } from "./types.js";

const DEFAULT_SAMPLE_IDS = "conv-30,conv-41,conv-42,conv-43,conv-44";

interface Options {
  data: string;
  db: string;
  out: string;
  dotnetDll: string;
  modelDir: string;
  domain: string;
  sampleIds: string;
  topK: number;
  answerContextK: number;
  limit: number;
  ingestLimit: number;
  answerModel: string;
  judgeModel: string;
  chatProvider: ChatProviderConfig;
  skipIngest: boolean;
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

function parseOptions(argv: string[]): Options {
  const chatProvider = resolveChatProvider(argv);
  return {
    data: getOption(argv, "data", "benchmark/locomo/data/locomo10.json"),
    db: getOption(argv, "db", "benchmark/locomo/results/locomo-dotnet-conversational.db"),
    out: getOption(argv, "out", "benchmark/locomo/results/locomo-dotnet-conversational-answer.json"),
    dotnetDll: getOption(argv, "dotnet-dll", "dotnet/src/PsmMemory.Cli/bin/Release/net10.0/PsmMemory.Cli.dll"),
    modelDir: getOption(argv, "model-dir", "psm-model/prod-memory/onnx-runtime/v2"),
    domain: getOption(argv, "domain", "conversational"),
    sampleIds: getOption(argv, "sample-ids", DEFAULT_SAMPLE_IDS),
    topK: Number(getOption(argv, "top-k", "5")),
    answerContextK: Number(getOption(argv, "answer-context-k", "5")),
    limit: Number(getOption(argv, "limit", "0")),
    ingestLimit: Number(getOption(argv, "ingest-limit", "0")),
    answerModel: getOption(argv, "answer-model", defaultAnswerModel(chatProvider)),
    judgeModel: getOption(argv, "judge-model", defaultJudgeModel(chatProvider)),
    chatProvider,
    skipIngest: argv.includes("--skip-ingest")
  };
}

// --- Persistent `dotnet ... serve` client ------------------------------------------------------

interface ServeResponse {
  id: string;
  ok: boolean;
  result?: Record<string, unknown>;
  error?: string;
}

class DotnetServeClient {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly pending = new Map<string, { resolve: (r: ServeResponse) => void; reject: (e: Error) => void }>();
  private nextId = 0;
  private readonly ready: Promise<void>;

  constructor(dotnetDll: string, dbPath: string, modelDir: string) {
    this.child = spawn("dotnet", [dotnetDll, "serve", "--db", dbPath, "--model-dir", modelDir], {
      stdio: ["pipe", "pipe", "pipe"]
    });
    const rl = createInterface({ input: this.child.stdout });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      let parsed: ServeResponse;
      try {
        parsed = JSON.parse(line) as ServeResponse;
      } catch (error) {
        process.stderr.write(`serve: failed to parse response line: ${line}\n`);
        return;
      }
      const waiter = this.pending.get(parsed.id);
      if (!waiter) return;
      this.pending.delete(parsed.id);
      waiter.resolve(parsed);
    });
    this.ready = new Promise((resolvePromise, rejectPromise) => {
      const errRl = createInterface({ input: this.child.stderr });
      errRl.on("line", (line) => {
        process.stderr.write(`[dotnet serve] ${line}\n`);
        if (line.includes("ready for NDJSON requests")) resolvePromise();
      });
      this.child.on("error", rejectPromise);
      this.child.on("exit", (code) => {
        if (code !== 0 && code !== null) rejectPromise(new Error(`dotnet serve exited early with code ${code}`));
      });
    });
  }

  async waitUntilReady(): Promise<void> {
    await this.ready;
  }

  private call(cmd: string, fields: Record<string, unknown>): Promise<ServeResponse> {
    const id = String(this.nextId++);
    const request = { id, cmd, ...fields };
    return new Promise((resolvePromise, rejectPromise) => {
      this.pending.set(id, { resolve: resolvePromise, reject: rejectPromise });
      this.child.stdin.write(`${JSON.stringify(request)}\n`);
    });
  }

  async remember(fields: Record<string, unknown>): Promise<Record<string, unknown>> {
    const response = await this.call("remember", fields);
    if (!response.ok) throw new Error(response.error ?? "remember failed");
    return response.result ?? {};
  }

  async recall(fields: Record<string, unknown>): Promise<Record<string, unknown>> {
    const response = await this.call("recall", fields);
    if (!response.ok) throw new Error(response.error ?? "recall failed");
    return response.result ?? {};
  }

  close(): void {
    this.child.stdin.end();
  }
}

// --- Ingest phase -------------------------------------------------------------------------------

interface IngestStats {
  seen: number;
  stored: number;
  ignored: number;
  failed: number;
  errors: Array<{ source: string; error: string }>;
}

async function ingestSample(
  client: DotnetServeClient,
  sample: LocomoSample,
  domain: string,
  stats: IngestStats,
  ingestLimit: number
): Promise<void> {
  const sampleId = String(sample.sample_id ?? "unknown");
  const userId = `locomo-${sampleId}`;
  const turns = flattenTurns(sample);
  for (const turn of turns) {
    if (ingestLimit > 0 && stats.seen >= ingestLimit) return;
    const diaId = String(turn.dia_id ?? "");
    const source = `${sampleId}:${diaId}`;
    stats.seen++;
    try {
      const result = await client.remember({
        llmResponse: `${turn.speaker ?? "Unknown"}: ${turn.text ?? ""}`.trim(),
        userId,
        domain,
        includeExistingMemories: false,
        source: {
          sourceKind: "locomo_turn",
          sourceId: diaId,
          sourceTimestamp: locomoSourceTimestamp(sample, turn.session)
        }
      });
      const action = String(result.action ?? "");
      if (action === "ignore") stats.ignored++;
      else if (Array.isArray(result.written) && result.written.length > 0) stats.stored++;
      else stats.ignored++;
    } catch (error) {
      stats.failed++;
      stats.errors.push({ source, error: error instanceof Error ? error.message : String(error) });
    }
    if (stats.seen % 50 === 0) {
      process.stdout.write(`ingested ${stats.seen} | stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}\n`);
    }
  }
}

// --- Answer + judge phase (methodology mirrors answer-evaluate.ts) ---------------------------

interface ContextItem {
  memory_id: string;
  table: string;
  content: string;
  score?: number;
  source_ids: string[];
}

interface AnswerRecord {
  sample_id: string;
  category: string;
  question: string;
  gold_answer: string;
  evidence: string[];
  retrieved_ids: string[];
  answer_context_ids: string[];
  hit_at_1: boolean;
  hit_at_k: boolean;
  answer_context_hit_at_k: boolean;
  context_items: ContextItem[];
  generated_answer: string;
  answer_evidence_ids: string[];
  answer_json_parse_error?: string;
  judgment: "correct" | "incorrect";
  score: number;
  judge_reasoning: string;
}

function extractContextItems(recallResult: Record<string, unknown>): ContextItem[] {
  const memories = Array.isArray(recallResult.memories) ? recallResult.memories : [];
  return memories
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map((item) => {
      const memory = (item.memory && typeof item.memory === "object" ? item.memory : {}) as Record<string, unknown>;
      const table = typeof item.table === "string" ? item.table : "episodic";
      const id = typeof item.id === "string" ? item.id : "";
      const content = typeof item.content === "string" ? item.content : "";
      const sourceId = typeof memory.sourceId === "string" ? memory.sourceId : undefined;
      return {
        memory_id: id,
        table,
        content,
        score: typeof item.score === "number" ? item.score : undefined,
        source_ids: sourceId ? [sourceId] : []
      };
    })
    .filter((item) => item.memory_id && item.content.trim());
}

function hitAt(evidence: string[], selected: string[], k: number): boolean {
  const selectedSet = new Set(selected.slice(0, k));
  return evidence.some((id) => selectedSet.has(id));
}

function renderContextForPrompt(items: ContextItem[]): string {
  return items
    .map((item, index) => {
      const sources = item.source_ids.length > 0 ? ` sources=${item.source_ids.join(",")}` : "";
      return `[${index + 1}] [${item.table}] id=${item.memory_id}${sources} ${item.content}`;
    })
    .join("\n");
}

interface GeneratedAnswer {
  answer: string;
  evidenceIds: string[];
  parseError?: string;
}

function cleanAnswer(value: string): string {
  let answer = value.trim();
  answer = answer.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const finalMatch = answer.match(/(?:final answer|answer)\s*:\s*([\s\S]*)/i);
  if (finalMatch?.[1]) answer = finalMatch[1].trim();
  const sentences = answer.split(/(?<=[.!?])\s+/).filter(Boolean);
  if (sentences.length > 3) answer = sentences.slice(-2).join(" ");
  return answer.trim();
}

function parseAnswerJson(value: string): GeneratedAnswer {
  const trimmed = value.trim();
  const json = trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed;
  try {
    const parsed = JSON.parse(json) as { answer?: unknown; evidence_ids?: unknown };
    const answer = typeof parsed.answer === "string" ? cleanAnswer(parsed.answer) : "";
    return {
      answer: answer || "No final answer generated.",
      evidenceIds: Array.isArray(parsed.evidence_ids) ? parsed.evidence_ids.map(String).filter(Boolean) : []
    };
  } catch (error) {
    return {
      answer: "No final answer generated.",
      evidenceIds: [],
      parseError: error instanceof Error ? error.message : String(error)
    };
  }
}

async function answerQuestion(options: Options, question: string, contextItems: ContextItem[]): Promise<GeneratedAnswer> {
  const context = renderContextForPrompt(contextItems);
  const content = await chatCompletion(options.chatProvider, options.answerModel, [
    {
      role: "system",
      content: "Answer a LOCOMO benchmark question using only the provided retrieved memories. Return JSON only with exactly this shape: {\"answer\":\"short final answer or I do not know.\",\"evidence_ids\":[\"D1:12\"]}. Do not include reasoning, markdown, citations outside JSON, or extra keys. For when/date questions, return the date or time phrase. For relationship/status questions, return the status. If the memories do not contain the answer, set answer to exactly \"I do not know.\" and evidence_ids to []. evidence_ids must only contain source IDs shown in the retrieved memories."
    },
    {
      role: "user",
      content: `Retrieved memories:\n${context}\n\nQuestion: ${question}\n\nReturn JSON only:`
    }
  ], 180, 0);
  return parseAnswerJson(content);
}

function parseJudgeJson(value: string): { correct: boolean; reasoning: string } {
  const trimmed = value.trim();
  const json = trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed;
  try {
    const parsed = JSON.parse(json) as { correct?: unknown; reasoning?: unknown };
    return {
      correct: parsed.correct === true || String(parsed.correct).toLowerCase() === "true",
      reasoning: typeof parsed.reasoning === "string" ? parsed.reasoning : ""
    };
  } catch {
    return { correct: /\btrue\b/i.test(trimmed) && !/\bfalse\b/i.test(trimmed), reasoning: trimmed };
  }
}

async function judgeAnswer(options: Options, question: string, goldAnswer: string, generatedAnswer: string): Promise<{ correct: boolean; reasoning: string }> {
  const content = await chatCompletion(options.chatProvider, options.judgeModel, [
    {
      role: "system",
      content: "You are judging a LOCOMO memory benchmark answer. Return JSON only: {\"correct\":true|false,\"reasoning\":\"short reason\"}. Mark correct when the generated answer is semantically consistent with the gold answer. Mark incorrect for missing, contradicted, or unsupported answers."
    },
    {
      role: "user",
      content: `Question: ${question}\nGold answer: ${goldAnswer}\nGenerated answer: ${generatedAnswer}`
    }
  ], 160, 0);
  const parsed = parseJudgeJson(content);
  return { correct: parsed.correct, reasoning: parsed.reasoning || content.trim() };
}

function summarize(records: AnswerRecord[], options: Options): Record<string, unknown> {
  const denom = records.length || 1;
  const byCategory: Record<string, { questions: number; answer_accuracy: number }> = {};
  for (const record of records) {
    const entry = byCategory[record.category] ?? { questions: 0, answer_accuracy: 0 };
    entry.questions++;
    entry.answer_accuracy += record.score;
    byCategory[record.category] = entry;
  }
  for (const entry of Object.values(byCategory)) entry.answer_accuracy = entry.answer_accuracy / (entry.questions || 1);

  return {
    metric: "LoCoMo LLM-as-judge answer accuracy (via real PsmMemory.Cli .NET product, domain=" + options.domain + ")",
    questions: records.length,
    answer_accuracy: records.reduce((sum, record) => sum + record.score, 0) / denom,
    evidence_hit_at_1: records.filter((record) => record.hit_at_1).length / denom,
    evidence_hit_at_k: records.filter((record) => record.hit_at_k).length / denom,
    answer_context_hit_at_k: records.filter((record) => record.answer_context_hit_at_k).length / denom,
    top_k: options.topK,
    answer_context_k: options.answerContextK,
    answer_model: options.answerModel,
    judge_model: options.judgeModel,
    llm_provider: options.chatProvider.provider,
    by_category: byCategory
  };
}

async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const samples = filterSamples(loadSamples(options.data), parseSampleIds(options.sampleIds));

  const client = new DotnetServeClient(resolve(options.dotnetDll), resolve(options.db), resolve(options.modelDir));
  await client.waitUntilReady();

  if (!options.skipIngest) {
    const stats: IngestStats = { seen: 0, stored: 0, ignored: 0, failed: 0, errors: [] };
    for (const sample of samples) {
      process.stdout.write(`ingesting ${sample.sample_id}...\n`);
      await ingestSample(client, sample, options.domain, stats, options.ingestLimit);
      if (options.ingestLimit > 0 && stats.seen >= options.ingestLimit) break;
    }
    process.stdout.write(`ingest complete: ${JSON.stringify(stats)}\n`);
  }

  const records: AnswerRecord[] = [];
  let processed = 0;
  for (const sample of samples) {
    const sampleId = String(sample.sample_id ?? "unknown");
    const userId = `locomo-${sampleId}`;
    for (const qa of sample.qa ?? []) {
      const evidence = (qa.evidence ?? []).map(String).filter(Boolean);
      if (evidence.length === 0) continue;
      const question = String(qa.question ?? "");
      const goldAnswer = String(qa.answer ?? "");
      const category = String(qa.category ?? "unknown");
      if (options.limit > 0 && processed >= options.limit) break;

      const recallResult = await client.recall({ question, userId, topK: options.topK, domain: options.domain });
      const contextItems = extractContextItems(recallResult).slice(0, options.topK);
      const retrievedIds = contextItems.flatMap((item) => item.source_ids);
      const hitAt1 = hitAt(evidence, retrievedIds, 1);
      const hitAtK = hitAt(evidence, retrievedIds, options.topK);
      const answerContextItems = contextItems.slice(0, options.answerContextK);
      const answerContextIds = answerContextItems.flatMap((item) => item.source_ids);
      const answerContextHitAtK = hitAt(evidence, answerContextIds, options.answerContextK);
      const answer = await answerQuestion(options, question, answerContextItems);
      const judgment = await judgeAnswer(options, question, goldAnswer, answer.answer);

      records.push({
        sample_id: sampleId,
        category,
        question,
        gold_answer: goldAnswer,
        evidence,
        retrieved_ids: retrievedIds,
        answer_context_ids: answerContextIds,
        hit_at_1: hitAt1,
        hit_at_k: hitAtK,
        answer_context_hit_at_k: answerContextHitAtK,
        context_items: contextItems,
        generated_answer: answer.answer,
        answer_evidence_ids: answer.evidenceIds,
        answer_json_parse_error: answer.parseError,
        judgment: judgment.correct ? "correct" : "incorrect",
        score: judgment.correct ? 1 : 0,
        judge_reasoning: judgment.reasoning
      });
      processed++;
      if (processed % 10 === 0) {
        const summary = summarize(records, options);
        process.stdout.write(`answered=${records.length} accuracy=${(summary.answer_accuracy as number).toFixed(4)}\n`);
        writeOutput(options.out, summary, records);
      }
    }
  }

  client.close();
  const summary = summarize(records, options);
  writeOutput(options.out, summary, records);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\nWrote ${options.out}\n`);
  return records.length === 0 ? 1 : 0;
}

function writeOutput(path: string, summary: Record<string, unknown>, records: AnswerRecord[]): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify({ summary, records }, null, 2), "utf8");
  renameSync(tmp, path);
}

process.exitCode = await main(process.argv.slice(2));
