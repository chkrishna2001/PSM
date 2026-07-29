/**
 * First real coding-domain analog to the LoCoMo benchmark: ingests a real, multi-agent (Claude Code,
 * Codex, pi), multi-session turn pool for the PSM project itself (benchmark/coding-locomo/data/
 * psm-project-v1-turns.json, 2026-07-04 -> 2026-07-29, 3558 turns) via the real PsmMemory.Cli .NET
 * product, then answers a small set of hand-authored, evidence-quote-backed QA pairs
 * (psm-project-v1.json) against it -- same DotnetServeClient / answer+judge methodology as
 * locomo-dotnet-benchmark.ts, adapted for this benchmark's flat single-conversation turn-pool shape
 * instead of LoCoMo's multi-sample/session structure. See docs/plans/psm-coding-locomo-benchmark.md
 * for the full design and data-provenance writeup.
 */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { chatCompletion, resolveChatProvider, defaultAnswerModel, defaultJudgeModel, type ChatProviderConfig } from "./cloudflare-ai.js";

interface Turn {
  text: string;
  timestamp: string;
  session_id: string;
  agent: string;
  source_path: string;
}

interface QaEvidence {
  agent: string;
  timestamp: string;
  quote: string;
}

interface Qa {
  id: string;
  category: string;
  question: string;
  answer: string;
  evidence: QaEvidence[];
}

interface BenchmarkData {
  conversation_id: string;
  qa: Qa[];
}

interface Options {
  turns: string;
  data: string;
  db: string;
  out: string;
  dotnetDll: string;
  modelDir: string;
  domain: string;
  userId: string;
  topK: number;
  answerContextK: number;
  ingestLimit: number;
  answerModel: string;
  judgeModel: string;
  chatProvider: ChatProviderConfig;
  skipIngest: boolean;
  skipAnswer: boolean;
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

function parseOptions(argv: string[]): Options {
  const chatProvider = resolveChatProvider(argv);
  return {
    turns: getOption(argv, "turns", "benchmark/coding-locomo/data/psm-project-v1-turns.json"),
    data: getOption(argv, "data", "benchmark/coding-locomo/data/psm-project-v1.json"),
    db: getOption(argv, "db", "benchmark/coding-locomo/results/coding-locomo.db"),
    out: getOption(argv, "out", "benchmark/coding-locomo/results/coding-locomo-answer.json"),
    dotnetDll: getOption(argv, "dotnet-dll", "dotnet/src/PsmMemory.Cli/bin/Release/net10.0/PsmMemory.Cli.dll"),
    modelDir: getOption(argv, "model-dir", "psm-model/prod-memory/gguf-runtime/v1"),
    domain: getOption(argv, "domain", "coding"),
    userId: getOption(argv, "user-id", "coding-locomo-psm-project-v1"),
    topK: Number(getOption(argv, "top-k", "5")),
    answerContextK: Number(getOption(argv, "answer-context-k", "5")),
    ingestLimit: Number(getOption(argv, "ingest-limit", "0")),
    answerModel: getOption(argv, "answer-model", defaultAnswerModel(chatProvider)),
    judgeModel: getOption(argv, "judge-model", defaultJudgeModel(chatProvider)),
    chatProvider,
    skipIngest: argv.includes("--skip-ingest"),
    skipAnswer: argv.includes("--skip-answer")
  };
}

// --- Persistent `dotnet ... serve` client (identical protocol to locomo-dotnet-benchmark.ts) ----

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
      } catch {
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

  async remember(fields: Record<string, unknown>): Promise<unknown[]> {
    const response = await this.call("remember", fields);
    if (!response.ok) throw new Error(response.error ?? "remember failed");
    return Array.isArray(response.result) ? response.result : [];
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

// --- Ingest phase --------------------------------------------------------------------------------

interface IngestStats {
  seen: number;
  stored: number;
  ignored: number;
  failed: number;
  errors: Array<{ source: string; error: string }>;
}

async function ingestTurns(client: DotnetServeClient, turns: Turn[], options: Options, stats: IngestStats): Promise<void> {
  for (const turn of turns) {
    if (options.ingestLimit > 0 && stats.seen >= options.ingestLimit) return;
    stats.seen++;
    const source = `${turn.agent}:${turn.session_id}:${turn.timestamp}`;
    try {
      const result = await client.remember({
        llmResponse: turn.text,
        userId: options.userId,
        domain: options.domain,
        includeExistingMemories: false,
        source: {
          sourceKind: "coding_locomo_turn",
          sourceId: source,
          sourceTimestamp: turn.timestamp
        }
      });
      for (const chunk of result) {
        const chunkRecord = (chunk && typeof chunk === "object" ? chunk : {}) as Record<string, unknown>;
        const action = String(chunkRecord.action ?? "");
        const written = chunkRecord.written;
        if (action !== "ignore" && Array.isArray(written) && written.length > 0) stats.stored++;
        else stats.ignored++;
      }
    } catch (error) {
      stats.failed++;
      stats.errors.push({ source, error: error instanceof Error ? error.message : String(error) });
    }
    if (stats.seen % 100 === 0) {
      process.stdout.write(`ingested ${stats.seen}/${turns.length} | stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}\n`);
    }
  }
}

// --- Answer + judge phase (methodology mirrors locomo-dotnet-benchmark.ts) -----------------------

interface ContextItem {
  memory_id: string;
  table: string;
  content: string;
  score?: number;
  source_id?: string;
  source_timestamp?: string;
}

interface AnswerRecord {
  id: string;
  category: string;
  question: string;
  gold_answer: string;
  retrieved_ids: string[];
  context_items: ContextItem[];
  generated_answer: string;
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
      const sourceTimestamp = typeof memory.sourceTimestamp === "string" ? memory.sourceTimestamp : undefined;
      return {
        memory_id: id,
        table,
        content,
        score: typeof item.score === "number" ? item.score : undefined,
        source_id: sourceId,
        source_timestamp: sourceTimestamp
      };
    })
    .filter((item) => item.memory_id && item.content.trim());
}

function renderContextForPrompt(items: ContextItem[]): string {
  return items
    .map((item, index) => {
      const sent = item.source_timestamp ? ` (message sent: ${item.source_timestamp})` : "";
      return `[${index + 1}] [${item.table}] id=${item.memory_id} ${item.content}${sent}`;
    })
    .join("\n");
}

interface GeneratedAnswer {
  answer: string;
  parseError?: string;
}

function cleanAnswer(value: string): string {
  let answer = value.trim();
  answer = answer.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const finalMatch = answer.match(/(?:final answer|answer)\s*:\s*([\s\S]*)/i);
  if (finalMatch?.[1]) answer = finalMatch[1].trim();
  return answer.trim();
}

function parseAnswerJson(value: string): GeneratedAnswer {
  const trimmed = value.trim();
  const json = trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed;
  try {
    const parsed = JSON.parse(json) as { answer?: unknown };
    const answer = typeof parsed.answer === "string" ? cleanAnswer(parsed.answer) : "";
    return { answer: answer || "No final answer generated." };
  } catch (error) {
    return { answer: "No final answer generated.", parseError: error instanceof Error ? error.message : String(error) };
  }
}

async function answerQuestion(options: Options, question: string, contextItems: ContextItem[]): Promise<GeneratedAnswer> {
  const context = renderContextForPrompt(contextItems);
  const content = await chatCompletion(options.chatProvider, options.answerModel, [
    {
      role: "system",
      content: "Answer a question about a real software project's history using only the provided retrieved memories. Return JSON only with exactly this shape: {\"answer\":\"short final answer or I do not know.\"}. Do not include reasoning, markdown, or extra keys. If the memories do not contain the answer, set answer to exactly \"I do not know.\""
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

async function judgeAnswer(options: Options, question: string, goldAnswer: string, generatedAnswer: string, category: string): Promise<{ correct: boolean; reasoning: string }> {
  const adversarialNote = category === "adversarial"
    ? " This is an ADVERSARIAL question with no real answer in the source data -- mark correct ONLY if the generated answer also says it does not know / has no answer, and mark incorrect if it confidently hallucinates a specific answer."
    : "";
  const content = await chatCompletion(options.chatProvider, options.judgeModel, [
    {
      role: "system",
      content: `You are judging an answer against a gold answer for a real coding-project memory benchmark. Return JSON only: {"correct":true|false,"reasoning":"short reason"}. Mark correct when the generated answer is semantically consistent with the gold answer.${adversarialNote}`
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
    metric: "coding-LoCoMo LLM-as-judge answer accuracy (via real PsmMemory.Cli .NET product, domain=" + options.domain + ")",
    questions: records.length,
    answer_accuracy: records.reduce((sum, record) => sum + record.score, 0) / denom,
    answer_model: options.answerModel,
    judge_model: options.judgeModel,
    llm_provider: options.chatProvider.provider,
    model_dir: options.modelDir,
    by_category: byCategory
  };
}

function writeOutput(path: string, summary: Record<string, unknown>, records: AnswerRecord[]): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify({ summary, records }, null, 2), "utf8");
  renameSync(tmp, path);
}

async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const turns: Turn[] = JSON.parse(readFileSync(resolve(options.turns), "utf8"));
  const data: BenchmarkData = JSON.parse(readFileSync(resolve(options.data), "utf8"));

  const client = new DotnetServeClient(resolve(options.dotnetDll), resolve(options.db), resolve(options.modelDir));
  await client.waitUntilReady();

  if (!options.skipIngest) {
    const stats: IngestStats = { seen: 0, stored: 0, ignored: 0, failed: 0, errors: [] };
    process.stdout.write(`ingesting ${turns.length} turns for user ${options.userId}...\n`);
    await ingestTurns(client, turns, options, stats);
    process.stdout.write(`ingest complete: ${JSON.stringify(stats)}\n`);
  }

  if (options.skipAnswer) {
    client.close();
    process.stdout.write("skip-answer set: ingest-only run complete, no QA/answer phase performed.\n");
    return 0;
  }

  const records: AnswerRecord[] = [];
  for (const qa of data.qa) {
    const recallResult = await client.recall({ question: qa.question, userId: options.userId, topK: options.topK, domain: options.domain });
    const contextItems = extractContextItems(recallResult).slice(0, options.topK);
    const retrievedIds = contextItems.map((item) => item.source_id).filter((id): id is string => Boolean(id));
    const answerContextItems = contextItems.slice(0, options.answerContextK);
    const answer = await answerQuestion(options, qa.question, answerContextItems);
    const judgment = await judgeAnswer(options, qa.question, qa.answer, answer.answer, qa.category);

    records.push({
      id: qa.id,
      category: qa.category,
      question: qa.question,
      gold_answer: qa.answer,
      retrieved_ids: retrievedIds,
      context_items: contextItems,
      generated_answer: answer.answer,
      answer_json_parse_error: answer.parseError,
      judgment: judgment.correct ? "correct" : "incorrect",
      score: judgment.correct ? 1 : 0,
      judge_reasoning: judgment.reasoning
    });
    const summary = summarize(records, options);
    process.stdout.write(`[${records.length}/${data.qa.length}] ${qa.id} (${qa.category}) -> ${judgment.correct ? "CORRECT" : "incorrect"} | running accuracy=${(summary.answer_accuracy as number).toFixed(3)}\n`);
    writeOutput(options.out, summary, records);
  }

  client.close();
  const summary = summarize(records, options);
  writeOutput(options.out, summary, records);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\nWrote ${options.out}\n`);
  return records.length === 0 ? 1 : 0;
}

process.exitCode = await main(process.argv.slice(2));
