import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { MemoryStore, rankMemories, type MemoryRecord, type RankedMemory } from "@psm-memory/sdk";
import { loadSamples, parseTags, tagValue } from "./common.js";

const defaultOpenRouterModel = "nvidia/nemotron-3-super-120b-a12b:free";

interface Options {
  data: string;
  db: string;
  out: string;
  topK: number;
  limit: number;
  answerModel: string;
  judgeModel: string;
  apiKey: string;
  baseUrl: string;
  resume: boolean;
  checkpointEvery: number;
}

interface AnswerRecord {
  sample_id: string;
  category: string;
  question: string;
  gold_answer: string;
  evidence: string[];
  selected_ids: string[];
  generated_answer: string;
  judgment: "correct" | "incorrect";
  score: number;
  judge_reasoning: string;
  answer_model: string;
  judge_model: string;
}

interface Output {
  summary: Record<string, unknown>;
  records: AnswerRecord[];
}

export async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  if (!options.apiKey) {
    throw new Error("OPENROUTER_API_KEY is required. Set it in Colab or pass --api-key.");
  }

  const existing = options.resume ? loadExisting(options.out) : { records: [] };
  const done = new Set(existing.records.map(recordKey));
  const records = [...existing.records];
  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  let processedThisRun = 0;

  try {
    for (const sample of samples) {
      const sampleId = String(sample.sample_id ?? "unknown");
      const userId = `locomo-${sampleId}`;
      const memories = store.selectMemories(userId, ["semantic", "episodic"], 10000);
      if (memories.length === 0) continue;

      for (const qa of sample.qa ?? []) {
        const evidence = (qa.evidence ?? []).map(String).filter(Boolean);
        if (evidence.length === 0) continue;
        const question = String(qa.question ?? "");
        const goldAnswer = String(qa.answer ?? "");
        const category = String(qa.category ?? "unknown");
        const key = `${sampleId}\n${category}\n${question}`;
        if (done.has(key)) continue;
        if (options.limit > 0 && processedThisRun >= options.limit) {
          writeOutput(options.out, summarize(records, options), records);
          return records.length === 0 ? 1 : 0;
        }

        const ranked = rankMemories(question, memories, options.topK);
        const selectedIds = ranked.map(locomoDiaId).filter(Boolean);
        const generatedAnswer = await answerQuestion(options, question, ranked);
        const judgment = await judgeAnswer(options, question, goldAnswer, generatedAnswer);

        records.push({
          sample_id: sampleId,
          category,
          question,
          gold_answer: goldAnswer,
          evidence,
          selected_ids: selectedIds,
          generated_answer: generatedAnswer,
          judgment: judgment.correct ? "correct" : "incorrect",
          score: judgment.correct ? 1 : 0,
          judge_reasoning: judgment.reasoning,
          answer_model: options.answerModel,
          judge_model: options.judgeModel
        });
        done.add(key);
        processedThisRun++;

        if (processedThisRun % options.checkpointEvery === 0) {
          writeOutput(options.out, summarize(records, options), records);
          process.stdout.write(`answered=${records.length} this_run=${processedThisRun} accuracy=${formatNumber(accuracy(records))}\n`);
        }
      }
    }
  } finally {
    store.close();
  }

  const summary = summarize(records, options);
  writeOutput(options.out, summary, records);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\nWrote ${options.out}\n`);
  return records.length === 0 ? 1 : 0;
}

async function answerQuestion(options: Options, question: string, memories: RankedMemory[]): Promise<string> {
  const context = memories.map((memory, index) => {
    const diaId = locomoDiaId(memory);
    const label = diaId ? `dia_id=${diaId}` : `memory_id=${memory.id}`;
    return `[${index + 1}] ${label} score=${memory.score}\n${memory.content}`;
  }).join("\n\n");
  const content = await chatCompletion(options, options.answerModel, [
    {
      role: "system",
      content: "Answer the user's LOCOMO benchmark question using only the provided retrieved memories. Be concise. If the memories do not contain the answer, say you do not know."
    },
    {
      role: "user",
      content: `Retrieved memories:\n${context}\n\nQuestion: ${question}\n\nAnswer:`
    }
  ], 256, 0);
  return content.trim();
}

async function judgeAnswer(options: Options, question: string, goldAnswer: string, generatedAnswer: string): Promise<{ correct: boolean; reasoning: string }> {
  const content = await chatCompletion(options, options.judgeModel, [
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
  return {
    correct: parsed.correct,
    reasoning: parsed.reasoning || content.trim()
  };
}

async function chatCompletion(options: Options, model: string, messages: Array<{ role: string; content: string }>, maxTokens: number, temperature: number): Promise<string> {
  const response = await fetch(`${options.baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "authorization": `Bearer ${options.apiKey}`,
      "content-type": "application/json"
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: maxTokens
    })
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Chat completion failed ${response.status}: ${body}`);
  }
  const data = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
  return data.choices?.[0]?.message?.content ?? "";
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
    return {
      correct: /\btrue\b/i.test(trimmed) && !/\bfalse\b/i.test(trimmed),
      reasoning: trimmed
    };
  }
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
  for (const entry of Object.values(byCategory)) {
    entry.answer_accuracy = entry.answer_accuracy / (entry.questions || 1);
  }

  return {
    metric: "LoCoMo LLM-as-judge answer accuracy",
    questions: records.length,
    answer_accuracy: records.reduce((sum, record) => sum + record.score, 0) / denom,
    top_k: options.topK,
    answer_model: options.answerModel,
    judge_model: options.judgeModel,
    by_category: byCategory
  };
}

function writeOutput(path: string, summary: Record<string, unknown>, records: AnswerRecord[]): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify({ summary, records } satisfies Output, null, 2), "utf8");
  renameSync(tmp, path);
}

function loadExisting(path: string): Output {
  if (!existsSync(path)) return { summary: {}, records: [] };
  const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<Output>;
  return {
    summary: parsed.summary ?? {},
    records: Array.isArray(parsed.records) ? parsed.records : []
  };
}

function recordKey(record: AnswerRecord): string {
  return `${record.sample_id}\n${record.category}\n${record.question}`;
}

function locomoDiaId(memory: MemoryRecord): string {
  return tagValue(parseTags(memory.tags), "locomo_dia_id");
}

function accuracy(records: AnswerRecord[]): number {
  const denom = records.length || 1;
  return records.reduce((sum, record) => sum + record.score, 0) / denom;
}

function formatNumber(value: number): string {
  return value.toFixed(4);
}

function parseOptions(argv: string[]): Options {
  const raw: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      raw[key] = next;
      i++;
    } else {
      raw[key] = true;
    }
  }

  return {
    data: stringOption(raw, "data", "benchmark/locomo/data/locomo10.json"),
    db: stringOption(raw, "db", "benchmark/locomo/results/locomo-psm-memory.db"),
    out: stringOption(raw, "out", "benchmark/locomo/results/locomo-answer-results.json"),
    topK: intOption(raw, "top-k", 50),
    limit: intOption(raw, "limit", 0),
    answerModel: stringOption(raw, "answer-model", process.env.LOCOMO_ANSWER_MODEL || defaultOpenRouterModel),
    judgeModel: stringOption(raw, "judge-model", process.env.LOCOMO_JUDGE_MODEL || defaultOpenRouterModel),
    apiKey: stringOption(raw, "api-key", process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || ""),
    baseUrl: stringOption(raw, "base-url", process.env.OPENROUTER_BASE_URL || process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1"),
    resume: raw.resume !== false && raw.resume !== "false",
    checkpointEvery: intOption(raw, "checkpoint-every", 10)
  };
}

function stringOption(options: Record<string, string | boolean>, key: string, fallback: string): string {
  const value = options[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intOption(options: Record<string, string | boolean>, key: string, fallback: number): number {
  const parsed = Number(options[key]);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

if (process.argv[1]?.endsWith("answer-evaluate.js")) {
  process.exitCode = await main(process.argv.slice(2));
}
