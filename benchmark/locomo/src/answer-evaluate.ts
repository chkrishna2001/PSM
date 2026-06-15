import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { homedir, platform } from "node:os";
import {
  defaultEmbeddingModel,
  MemoryStore,
  NodeLlamaRuntime,
  PsmModelRuntime,
  PsmService,
  TransformersEmbeddingRuntime,
  type ContextItem,
  type MemoryRecord
} from "@psm-memory/sdk";
import { loadSamples, parseTags, tagValue } from "./common.js";

const defaultOpenRouterModel = "nvidia/nemotron-3-super-120b-a12b:free";
const defaultPsmModelName = "psm-memory-qwen-1.5b-q4_k_m.gguf";

interface Options {
  data: string;
  db: string;
  out: string;
  topK: number;
  psmContextTopK: number;
  answerContextK: number;
  limit: number;
  psmModel: string;
  psmCheckpoint: string;
  psmPython: string;
  psmDevice: string;
  repoRoot: string;
  psmContextSize: number;
  psmGpu: "auto" | "cuda" | "vulkan" | "metal";
  psmGpuLayers: "auto" | "max" | number;
  embeddingModel: string;
  noEmbeddings: boolean;
  answerModel: string;
  judgeModel: string;
  apiKey: string;
  baseUrl: string;
  resume: boolean;
  checkpointEvery: number;
  requestDelayMs: number;
  requestMaxRetries: number;
  debugOut: string;
}

interface BenchmarkContextItem extends ContextItem {
  memory_id?: string;
  score?: number;
  source_ids?: string[];
}

interface AnswerRecord {
  sample_id: string;
  category: string;
  question: string;
  gold_answer: string;
  evidence: string[];
  recall_plan?: Record<string, unknown>;
  candidate_memory_ids: string[];
  retrieved_memory_ids: string[];
  retrieved_ids: string[];
  answer_context_memory_ids: string[];
  answer_context_ids: string[];
  hit_at_1: boolean;
  hit_at_k: boolean;
  answer_context_hit_at_k: boolean;
  psm_context_items: BenchmarkContextItem[];
  answer_context_items: BenchmarkContextItem[];
  gold_evidence_present_in_top_k: boolean;
  gold_evidence_used_in_answer_context: boolean;
  psm_context_parse_error?: string;
  psm_context_reasoning?: string;
  psm_context_raw_model_json?: string;
  generated_answer: string;
  answer_evidence_ids: string[];
  answer_raw_model_json?: string;
  answer_json_parse_error?: string;
  judgment: "correct" | "incorrect";
  score: number;
  judge_reasoning: string;
  failure_bucket?: string;
  answer_model: string;
  judge_model: string;
}

interface Output {
  summary: Record<string, unknown>;
  records: AnswerRecord[];
}

export async function main(argv: string[]): Promise<number> {
  const options = parseOptions(argv);
  const answerableOnly = argv.includes("--answerable-only");
  if (!options.apiKey) {
    throw new Error("OPENROUTER_API_KEY is required. Set it in Colab or pass --api-key.");
  }

  const existing = options.resume ? loadExisting(options.out) : { records: [] };
  const done = new Set(existing.records.map(recordKey));
  const records = [...existing.records];
  const samples = loadSamples(options.data);
  const store = new MemoryStore(options.db);
  const service = createPsmService(store, options);
  let processedThisRun = 0;

  try {
    for (const sample of samples) {
      const sampleId = String(sample.sample_id ?? "unknown");
      const userId = `locomo-${sampleId}`;
      const memories = store.selectMemories(userId, ["semantic", "episodic"], 10000);
      if (memories.length === 0) continue;
      const ingestedEvidenceIds = new Set(memories.flatMap(memoryEvidenceIds));

      for (const qa of sample.qa ?? []) {
        const evidence = (qa.evidence ?? []).map(String).filter(Boolean);
        if (evidence.length === 0) continue;
        if (answerableOnly && !evidence.some((id) => ingestedEvidenceIds.has(id))) continue;
        const question = String(qa.question ?? "");
        const goldAnswer = String(qa.answer ?? "");
        const category = String(qa.category ?? "unknown");
        const key = `${sampleId}\n${category}\n${question}`;
        if (done.has(key)) continue;
        if (options.limit > 0 && processedThisRun >= options.limit) {
          writeOutput(options.out, summarize(records, options), records);
          writeDebugReport(options.debugOut, records);
          return records.length === 0 ? 1 : 0;
        }

        const psmContext = await service.context({ prompt: question, userId, topK: options.psmContextTopK });
        const contextItems = extractBenchmarkContextItems(psmContext).slice(0, options.topK);
        const candidateMemories = extractMemoryContext(psmContext);
        const retrievedIds = contextItems.flatMap((item) => item.source_ids ?? []);
        const retrievedMemoryIds = contextItems.map((item) => `${item.table}:${item.memory_id ?? item.id ?? ""}`).filter((id) => !id.endsWith(":"));
        const candidateMemoryIds = candidateMemories.map((item) => `${item.table}:${item.id}`);
        const hitAt1 = hitAt(evidence, retrievedIds, 1);
        const hitAtK = hitAt(evidence, retrievedIds, options.topK);
        const answerContextItems = contextItems.slice(0, options.answerContextK);
        const answerContextIds = answerContextItems.flatMap((item) => item.source_ids ?? []);
        const answerContextMemoryIds = answerContextItems.map((item) => `${item.table}:${item.memory_id ?? item.id ?? ""}`).filter((id) => !id.endsWith(":"));
        const answerContextHitAtK = hitAt(evidence, answerContextIds, options.answerContextK);
        const answer = await answerQuestion(options, question, answerContextItems);
        const judgment = await judgeAnswer(options, question, goldAnswer, answer.answer);
        const evidenceInMemory = memories.some((memory) => memoryEvidenceIds(memory).some((id) => evidence.includes(id)));

        records.push({
          sample_id: sampleId,
          category,
          question,
          gold_answer: goldAnswer,
          evidence,
          recall_plan: asRecord(psmContext.recall_plan),
          candidate_memory_ids: candidateMemoryIds,
          retrieved_memory_ids: retrievedMemoryIds,
          retrieved_ids: retrievedIds,
          answer_context_memory_ids: answerContextMemoryIds,
          answer_context_ids: answerContextIds,
          hit_at_1: hitAt1,
          hit_at_k: hitAtK,
          answer_context_hit_at_k: answerContextHitAtK,
          psm_context_items: contextItems,
          answer_context_items: answerContextItems,
          gold_evidence_present_in_top_k: hitAtK,
          gold_evidence_used_in_answer_context: answerContextHitAtK,
          psm_context_parse_error: typeof psmContext.context_parse_error === "string" ? psmContext.context_parse_error : undefined,
          psm_context_reasoning: typeof psmContext.context_reasoning === "string" ? psmContext.context_reasoning : undefined,
          psm_context_raw_model_json: typeof psmContext.context_raw_model_json === "string" ? psmContext.context_raw_model_json : undefined,
          generated_answer: answer.answer,
          answer_evidence_ids: answer.evidenceIds,
          answer_raw_model_json: answer.raw,
          answer_json_parse_error: answer.parseError,
          judgment: judgment.correct ? "correct" : "incorrect",
          score: judgment.correct ? 1 : 0,
          judge_reasoning: judgment.reasoning,
          failure_bucket: classifyFailure({
            correct: judgment.correct,
            question,
            goldAnswer,
            generatedAnswer: answer.answer,
            evidenceInMemory,
            hitAtK: answerContextHitAtK,
            contextItems,
            judgeReasoning: judgment.reasoning
          }),
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
  writeDebugReport(options.debugOut, records);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\nWrote ${options.out}\n`);
  if (options.debugOut) process.stdout.write(`Wrote ${options.debugOut}\n`);
  return records.length === 0 ? 1 : 0;
}

interface GeneratedAnswer {
  answer: string;
  evidenceIds: string[];
  raw?: string;
  parseError?: string;
}

async function answerQuestion(options: Options, question: string, contextItems: BenchmarkContextItem[]): Promise<GeneratedAnswer> {
  const context = renderContextForPrompt(contextItems);
  const content = await chatCompletion(options, options.answerModel, [
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

function renderContextForPrompt(contextItems: BenchmarkContextItem[]): string {
  const context = contextItems.map((item, index) => {
    const table = typeof item.table === "string" ? item.table : "memory";
    const content = typeof item.content === "string" ? item.content : "";
    const id = typeof item.memory_id === "string" && item.memory_id ? ` id=${item.memory_id}` : "";
    const sources = item.source_ids && item.source_ids.length > 0 ? ` sources=${item.source_ids.join(",")}` : "";
    return `[${index + 1}] [${table}]${id}${sources} ${content}`;
  }).join("\n");
  return context;
}

function parseAnswerJson(value: string): GeneratedAnswer {
  const trimmed = value.trim();
  const json = trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed;
  try {
    const parsed = JSON.parse(json) as { answer?: unknown; evidence_ids?: unknown };
    const answer = typeof parsed.answer === "string" ? cleanAnswer(parsed.answer) : "";
    return {
      answer: answer || "No final answer generated.",
      evidenceIds: Array.isArray(parsed.evidence_ids) ? parsed.evidence_ids.map(String).filter(Boolean) : [],
      raw: trimmed
    };
  } catch (error) {
    return {
      answer: "No final answer generated.",
      evidenceIds: [],
      raw: trimmed,
      parseError: error instanceof Error ? error.message : String(error)
    };
  }
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
  let lastError = "";
  for (let attempt = 0; attempt <= options.requestMaxRetries; attempt++) {
    if (options.requestDelayMs > 0) await sleep(options.requestDelayMs);
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
    if (response.ok) {
      const data = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
      return data.choices?.[0]?.message?.content ?? "";
    }
    const body = await response.text();
    lastError = `Chat completion failed ${response.status}: ${body}`;
    if (response.status !== 429 || attempt >= options.requestMaxRetries) break;
    const waitMs = rateLimitWaitMs(response, body, attempt);
    process.stderr.write(`OpenRouter rate limited; retrying in ${Math.round(waitMs / 1000)}s (attempt ${attempt + 1}/${options.requestMaxRetries})\n`);
    await sleep(waitMs);
  }
  throw new Error(lastError);
}

function rateLimitWaitMs(response: any, body: string, attempt: number): number {
  const retryAfter = Number(response.headers?.get("retry-after"));
  if (Number.isFinite(retryAfter) && retryAfter > 0) return clamp(retryAfter * 1000, 1000, 120000);
  try {
    const parsed = JSON.parse(body) as { error?: { metadata?: { headers?: Record<string, string> } } };
    const reset = Number(parsed.error?.metadata?.headers?.["X-RateLimit-Reset"]);
    if (Number.isFinite(reset) && reset > 0) {
      const wait = reset - Date.now() + 1000;
      if (wait > 0) return clamp(wait, 1000, 120000);
    }
  } catch {
    // Fall through to exponential backoff.
  }
  return clamp(3000 * 2 ** attempt, 3000, 120000);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

function cleanAnswer(value: string): string {
  let answer = value.trim();
  answer = answer.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  answer = answer.replace(/^(we need to answer|let'?s answer|analysis|reasoning|thought process)\s*:?.*?\n+/is, "").trim();
  const finalMatch = answer.match(/(?:final answer|answer)\s*:\s*([\s\S]*)/i);
  if (finalMatch?.[1]) answer = finalMatch[1].trim();
  const sentences = answer.split(/(?<=[.!?])\s+/).filter(Boolean);
  if (sentences.length > 3) answer = sentences.slice(-2).join(" ");
  return answer.trim();
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
    evidence_hit_at_1: records.filter((record) => record.hit_at_1).length / denom,
    evidence_hit_at_k: records.filter((record) => record.hit_at_k).length / denom,
    answer_context_hit_at_k: records.filter((record) => record.answer_context_hit_at_k).length / denom,
    top_k: options.topK,
    psm_context_top_k: options.psmContextTopK,
    answer_context_k: options.answerContextK,
    psm_model: options.psmCheckpoint || options.psmModel,
    embedding_model: options.noEmbeddings ? null : options.embeddingModel,
    answer_model: options.answerModel,
    judge_model: options.judgeModel,
    db: options.db,
    generated_at: new Date().toISOString(),
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

function writeDebugReport(path: string, records: AnswerRecord[]): void {
  if (!path) return;
  mkdirSync(dirname(path), { recursive: true });
  const lines = [
    "# LOCOMO Answer Evaluation Debug Report",
    "",
    "| # | Sample | Category | Bucket | Hit@K | Answer Hit@K | Judgment | Question | Gold | Answer | Evidence | Retrieved Sources | Answer Context Sources | Answer Evidence | Retrieved Memories | Answer Context | Parse Error | Judge Reasoning |",
    "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ...records.slice(0, 50).map((record, index) => [
      index + 1,
      escapeCell(record.sample_id),
      escapeCell(record.category),
      escapeCell(record.failure_bucket ?? ""),
      record.hit_at_k ? "yes" : "no",
      record.answer_context_hit_at_k ? "yes" : "no",
      record.judgment,
      escapeCell(record.question),
      escapeCell(record.gold_answer),
      escapeCell(record.generated_answer),
      escapeCell(record.evidence.join(", ")),
      escapeCell(unique(record.retrieved_ids ?? []).join(", ")),
      escapeCell(unique(record.answer_context_ids ?? []).join(", ")),
      escapeCell(unique(record.answer_evidence_ids ?? []).join(", ")),
      escapeCell((record.retrieved_memory_ids ?? []).join(", ")),
      escapeCell((record.answer_context_items ?? record.psm_context_items ?? []).map((item) => item.content).join(" / ")),
      escapeCell(record.answer_json_parse_error ?? ""),
      escapeCell(record.judge_reasoning)
    ].join(" | ").replace(/^/, "| ").replace(/$/, " |"))
  ];
  writeFileSync(path, lines.join("\n"), "utf8");
}

function recordKey(record: AnswerRecord): string {
  return `${record.sample_id}\n${record.category}\n${record.question}`;
}

function accuracy(records: AnswerRecord[]): number {
  const denom = records.length || 1;
  return records.reduce((sum, record) => sum + record.score, 0) / denom;
}

interface RecallMemory {
  table: "episodic" | "semantic" | "archival";
  id: string;
  content: string;
  score?: number;
  created_at?: string;
  source_id?: string;
  source_timestamp?: string;
  resolved_time?: string;
  metadata?: Record<string, unknown>;
}

function extractBenchmarkContextItems(result: Record<string, unknown>): BenchmarkContextItem[] {
  const contextItems = Array.isArray(result.context_items) ? result.context_items : [];
  const candidates = extractMemoryContext(result);
  return contextItems
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map((item, index) => {
      const candidate = candidates[index];
      const table = recallTable(item.table ?? candidate?.table);
      const memoryId = typeof item.memory_id === "string" ? item.memory_id : candidate?.id;
      return {
        id: typeof item.id === "string" ? item.id : memoryId ? `${table}:${memoryId}` : undefined,
        memory_id: memoryId,
        table,
        content: typeof item.content === "string" ? item.content : "",
        reason: typeof item.reason === "string" ? item.reason : undefined,
        score: typeof item.score === "number" ? item.score : candidate?.score,
        source_ids: unique([...sourceIdsFromContextItem(item), ...(candidate ? sourceIdsFromMetadata(candidate.metadata) : [])]),
        source_timestamp: typeof item.source_timestamp === "string" ? item.source_timestamp : candidate?.source_timestamp,
        saved_at: typeof item.saved_at === "string" ? item.saved_at : candidate?.created_at,
        resolved_time: typeof item.resolved_time === "string" ? item.resolved_time : candidate?.resolved_time
      };
    })
    .filter((item) => item.content.trim().length > 0);
}

function extractMemoryContext(result: Record<string, unknown>): RecallMemory[] {
  return extractRecallMemories({ memories: result.memory_context });
}

function extractRecallMemories(result: Record<string, unknown>): RecallMemory[] {
  const memories = Array.isArray(result.memories) ? result.memories : [];
  return memories
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map((item) => ({
      table: recallTable(item.table),
      id: typeof item.id === "string" ? item.id : "",
      content: typeof item.content === "string" ? item.content : "",
      score: typeof item.score === "number" ? item.score : undefined,
      created_at: typeof item.created_at === "string" ? item.created_at : undefined,
      source_id: typeof item.source_id === "string" ? item.source_id : undefined,
      source_timestamp: typeof item.source_timestamp === "string" ? item.source_timestamp : undefined,
      resolved_time: typeof item.resolved_time === "string" ? item.resolved_time : undefined,
      metadata: asRecord(item.metadata)
    }))
    .filter((item) => item.id && item.content.trim());
}

function renderExactContextItems(memories: RecallMemory[], topK: number): BenchmarkContextItem[] {
  return memories.slice(0, topK).map((memory) => ({
    id: `${memory.table}:${memory.id}`,
    memory_id: memory.id,
    table: memory.table,
    content: memory.content,
    score: memory.score,
    source_ids: sourceIdsFromMetadata(memory.metadata),
    reason: memory.score == null ? "Selected by PSM recall." : `Selected by PSM recall, score ${memory.score}.`
  }));
}

function sourceIdsFromMetadata(metadata: Record<string, unknown> | undefined): string[] {
  const tags = Array.isArray(metadata?.tags) ? metadata.tags.map(String) : [];
  return evidenceIdsFromTags(tags);
}

function sourceIdsFromContextItem(item: Record<string, unknown>): string[] {
  const ids = new Set<string>();
  if (Array.isArray(item.source_ids)) {
    for (const id of item.source_ids.map(String).filter(Boolean)) ids.add(id);
  }
  if (typeof item.source_id === "string") {
    for (const id of evidenceIdsFromSourceId(item.source_id)) ids.add(id);
  }
  return [...ids];
}

function memoryEvidenceIds(memory: MemoryRecord): string[] {
  return evidenceIdsFromTags(parseTags(memory.tags));
}

function evidenceIdsFromTags(tags: string[]): string[] {
  const ids = new Set<string>();
  const diaId = tagValue(tags, "locomo_dia_id");
  if (diaId) ids.add(diaId);
  for (const key of ["related_dia_ids", "locomo_related_dia_ids"]) {
    const value = tagValue(tags, key);
    for (const id of value.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean)) ids.add(id);
  }
  return [...ids];
}

function evidenceIdsFromSourceId(sourceId: string): string[] {
  const match = sourceId.match(/(?:^|:)(D\d+:\d+)$/);
  return match ? [match[1]] : [];
}

function hitAt(evidence: string[], selected: string[], k: number): boolean {
  const selectedSet = new Set(selected.slice(0, k));
  return evidence.some((id) => selectedSet.has(id));
}

function classifyFailure(input: {
  correct: boolean;
  question: string;
  goldAnswer: string;
  generatedAnswer: string;
  evidenceInMemory: boolean;
  hitAtK: boolean;
  contextItems: BenchmarkContextItem[];
  judgeReasoning: string;
}): string {
  if (input.correct) return "";
  const text = `${input.question} ${input.goldAnswer} ${input.generatedAnswer} ${input.contextItems.map((item) => item.content).join(" ")}`.toLowerCase();
  if (/image|photo|picture|caption|shown|seen/.test(text)) return "image_context_missing";
  if (/\byesterday\b|\btoday\b|\btomorrow\b|\blast\b|\bnext\b/.test(text)) return "ambiguous_relative_date";
  if (/\bspeaker\b|\bhe\b|\bshe\b|\bthey\b|\bher\b|\bhis\b|\btheir\b/.test(input.question.toLowerCase())) return "speaker_confusion";
  if (/judge|evaluation|gold/.test(input.judgeReasoning.toLowerCase())) return "judge_error";
  if (!input.evidenceInMemory) return "missing_memory";
  if (!input.hitAtK) return "retrieval_miss";
  if (input.contextItems.length === 0) return "bad_context_selection";
  if (/i do not know|unknown|not enough|cannot determine/i.test(input.generatedAnswer)) return "bad_context_selection";
  return "answer_model_error";
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function createPsmService(store: MemoryStore, options: Options): PsmService {
  const runtime = options.psmCheckpoint
    ? new PsmModelRuntime({
        checkpoint: resolve(options.repoRoot, options.psmCheckpoint),
        python: options.psmPython,
        repoRoot: options.repoRoot,
        device: options.psmDevice,
        outputFormat: "tagged"
      })
    : new NodeLlamaRuntime({
        modelPath: options.psmModel,
        contextSize: options.psmContextSize,
        gpu: options.psmGpu,
        gpuLayers: options.psmGpuLayers,
        log: (message) => process.stderr.write(`${message}\n`)
      });
  const embeddings = options.noEmbeddings ? undefined : {
    model: options.embeddingModel,
    runtime: new TransformersEmbeddingRuntime({
      model: options.embeddingModel,
      cacheDir: join(modelCacheBaseDir(), "hf")
    })
  };
  return new PsmService(store, runtime, embeddings);
}

function contextPrompt(question: string, memories: unknown[]): string {
  return [
    question,
    "",
    "Select memory context that helps answer this LOCOMO question.",
    "The answerer will only see the context items you return, so include specific names, dates, places, relationships, and facts when relevant.",
    `Candidate memory count: ${memories.length}`
  ].join("\n");
}

function recallTable(value: unknown): RecallMemory["table"] {
  return value === "episodic" || value === "semantic" || value === "archival" ? value : "episodic";
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function escapeCell(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\r?\n/g, " ").trim();
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
    topK: intOption(raw, "top-k", 5),
    psmContextTopK: intOption(raw, "psm-context-top-k", intOption(raw, "top-k", 5)),
    answerContextK: intOption(raw, "answer-context-k", Math.min(5, intOption(raw, "top-k", 5))),
    limit: intOption(raw, "limit", 0),
    psmModel: stringOption(raw, "psm-model", process.env.PSM_MEMORY_MODEL || defaultModelPath()),
    psmCheckpoint: stringOption(raw, "checkpoint", process.env.PSM_CHECKPOINT || ""),
    psmPython: stringOption(raw, "python", process.env.PSM_PYTHON || (platform() === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python")),
    psmDevice: stringOption(raw, "device", process.env.PSM_DEVICE || "auto"),
    repoRoot: stringOption(raw, "repo-root", process.cwd()),
    psmContextSize: intOption(raw, "psm-context-size", 4096),
    psmGpu: stringOption(raw, "psm-gpu", "auto") as Options["psmGpu"],
    psmGpuLayers: parseGpuLayers(stringOption(raw, "psm-gpu-layers", "auto")),
    embeddingModel: stringOption(raw, "embedding-model", process.env.PSM_MEMORY_EMBEDDING_MODEL || defaultEmbeddingModel),
    noEmbeddings: raw["no-embeddings"] === true || raw["no-embeddings"] === "true",
    answerModel: stringOption(raw, "answer-model", process.env.LOCOMO_ANSWER_MODEL || defaultOpenRouterModel),
    judgeModel: stringOption(raw, "judge-model", process.env.LOCOMO_JUDGE_MODEL || defaultOpenRouterModel),
    apiKey: stringOption(raw, "api-key", process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || ""),
    baseUrl: stringOption(raw, "base-url", process.env.OPENROUTER_BASE_URL || process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1"),
    resume: raw.resume !== false && raw.resume !== "false",
    checkpointEvery: intOption(raw, "checkpoint-every", 10),
    requestDelayMs: intOption(raw, "request-delay-ms", intOption(raw, "openrouter-delay-ms", Number(process.env.LOCOMO_REQUEST_DELAY_MS ?? 1500))),
    requestMaxRetries: intOption(raw, "request-max-retries", Number(process.env.LOCOMO_REQUEST_MAX_RETRIES ?? 6)),
    debugOut: stringOption(raw, "debug-out", defaultDebugOut(stringOption(raw, "out", "benchmark/locomo/results/locomo-answer-results.json")))
  };
}

function parseGpuLayers(value: string): Options["psmGpuLayers"] {
  if (value === "auto" || value === "max") return value;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : "auto";
}

function modelCacheBaseDir(): string {
  return process.env.PSM_MEMORY_HOME || dirname(defaultModelPath());
}

function defaultModelPath(): string {
  const explicitHome = process.env.PSM_MEMORY_HOME;
  if (explicitHome?.trim()) return join(explicitHome, defaultPsmModelName);
  if (platform() === "win32") {
    return join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "psm-memory", "models", defaultPsmModelName);
  }
  return join(process.env.XDG_CACHE_HOME || join(homedir(), ".cache"), "psm-memory", "models", defaultPsmModelName);
}

function defaultDebugOut(out: string): string {
  return out.replace(/(?:\.json)?$/i, "-debug.md");
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
