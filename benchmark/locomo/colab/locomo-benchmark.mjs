import { copyFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import {
  MemoryStore,
  NodeLlamaRuntime,
  PsmService,
  rankMemories
} from "@psm-memory/sdk";

const defaultOpenRouterModel = "nvidia/nemotron-3-super-120b-a12b:free";

const command = process.argv[2] ?? "help";
const args = parseArgs(process.argv.slice(3));

if (command === "ingest") {
  process.exitCode = await ingest(args);
} else if (command === "evaluate") {
  process.exitCode = evaluate(args);
} else if (command === "answer-evaluate") {
  process.exitCode = await answerEvaluate(args);
} else if (command === "report") {
  process.exitCode = report(args);
} else {
  console.log(`Usage:
  node locomo-benchmark.mjs ingest --data <locomo10.json> --db <db> --model <gguf> [--limit n] [--batch-size n] [--progress progress.json] [--checkpoint-dir dir]
  node locomo-benchmark.mjs evaluate --data <locomo10.json> --db <db> --out <results.json> [--top-k n]
  node locomo-benchmark.mjs answer-evaluate --data <locomo10.json> --db <db> --out <answer-results.json> [--top-k n]
  node locomo-benchmark.mjs report --psm <locomo-results.json> --baselines <memory-tools.json> --out <comparison.md>
`);
}

async function ingest(options) {
  const dataPath = stringOption(options, "data", "/content/PSM/benchmark/locomo/data/locomo10.json");
  const dbPath = stringOption(options, "db", "/content/locomo/results/locomo-psm-memory.db");
  const modelPath = stringOption(options, "model", "/content/psm-memory-cache/psm-memory-qwen-1.5b-q4_k_m.gguf");
  const limit = intOption(options, "limit", 100);
  const batchSize = intOption(options, "batch-size", 10);
  const offset = intOption(options, "offset", 0);
  const progressPath = stringOption(options, "progress", "");
  const summaryPath = stringOption(options, "summary", "/content/locomo/results/ingest-summary.json");
  const checkpointDir = stringOption(options, "checkpoint-dir", "");
  const contextSize = intOption(options, "context-size", 4096);
  const windowSize = intOption(options, "window-size", 2);
  const userPrefix = stringOption(options, "user-prefix", "locomo");
  const records = loadSamples(dataPath).flatMap((sample) => {
    const sampleId = String(sample.sample_id ?? "unknown");
    const userId = `${userPrefix}-${sampleId}`;
    const turns = flattenTurns(sample);
    return turns.map((turn, sampleOrdinal) => ({
      sampleId,
      userId,
      sample,
      turns,
      turn,
      sampleOrdinal
    }));
  });
  const progress = loadProgress(progressPath);
  const progressIndex = Number.isInteger(progress.next_index) ? progress.next_index : 0;
  const checkpointDbPath = checkpointDir ? join(checkpointDir, basename(dbPath)) : "";
  const hasResumableDb = existsSync(dbPath) || (checkpointDbPath ? existsSync(checkpointDbPath) : false);
  if (progressIndex > 0 && checkpointDir && !hasResumableDb) {
    console.warn(`Ignoring progress next_index=${progressIndex} because no DB checkpoint exists at ${checkpointDbPath}`);
  }
  const startIndex = Math.max(offset, progressIndex > 0 && (!checkpointDir || hasResumableDb) ? progressIndex : 0);
  const endIndex = limit > 0 ? Math.min(records.length, startIndex + limit) : records.length;
  mkdirSync(dirname(dbPath), { recursive: true });
  if (checkpointDir) mkdirSync(checkpointDir, { recursive: true });

  const store = new MemoryStore(dbPath);
  store.initializeSchema();
  const runtime = createLocomoIngestRuntime(new NodeLlamaRuntime({
    modelPath,
    contextSize,
    gpu: "auto",
    gpuLayers: "auto",
    log: (message) => console.error(message)
  }));
  const service = new PsmService(store, runtime);

  const stats = {
    data: dataPath,
    db: dbPath,
    model: modelPath,
    limit,
    batch_size: batchSize,
    offset,
    progress: progressPath || null,
    summary: summaryPath,
    checkpoint_dir: checkpointDir || null,
    total_records: records.length,
    start_index: startIndex,
    end_index: endIndex,
    next_index: startIndex,
    seen: 0,
    stored: 0,
    ignored: 0,
    failed: 0,
    started_at: new Date().toISOString(),
    ended_at: null,
    errors: []
  };

  try {
    if (startIndex >= records.length) {
      checkpoint(dbPath, checkpointDir, progressPath, stats, startIndex);
      return finish(store, stats, summaryPath);
    }

    for (let index = startIndex; index < endIndex; index++) {
      const { sampleId, userId, sample, turns, turn, sampleOrdinal } = records[index];
      const diaId = String(turn.dia_id ?? "");
      const source = `${sampleId}:${diaId || sampleOrdinal}`;
      stats.seen++;
      try {
        const result = await service.remember({
          userId,
          llmResponse: buildLocomoRememberText({ sample, turns, index: sampleOrdinal, windowSize }),
          userMessage: `${turn.speaker ?? "Unknown"} said: ${turn.text ?? ""}`.trim(),
          source: {
            source_kind: "locomo_turn",
            source_id: source,
            source_timestamp: locomoSourceTimestamp(sample, turn.session),
            source_label: `LOCOMO ${sampleId} ${diaId || sampleOrdinal}`
          },
          includeExistingMemories: false,
          extraTags: [
            `locomo_sample_id:${sampleId}`,
            `locomo_dia_id:${diaId}`,
            `locomo_speaker:${turn.speaker ?? ""}`,
            `locomo_session:${turn.session ?? ""}`
          ]
        });
        recordRememberResult(stats, source, result);
      } catch (error) {
        stats.failed++;
        const message = error instanceof Error ? error.message : String(error);
        stats.errors.push({ source, error: message });
        store.insertDecision(userId, source, "error", "error", message, JSON.stringify({ error: message }));
      }
      stats.next_index = index + 1;
      if (stats.seen % batchSize === 0) {
        checkpoint(dbPath, checkpointDir, progressPath, stats, stats.next_index);
        console.log(`ingested=${stats.next_index}/${records.length} run_seen=${stats.seen} stored=${stats.stored} ignored=${stats.ignored} failed=${stats.failed}`);
      }
    }
    checkpoint(dbPath, checkpointDir, progressPath, stats, stats.next_index);
    return finish(store, stats, summaryPath);
  } finally {
    store.close();
  }
}

function evaluate(options) {
  const dataPath = stringOption(options, "data", "/content/PSM/benchmark/locomo/data/locomo10.json");
  const dbPath = stringOption(options, "db", "/content/locomo/results/locomo-psm-memory.db");
  const outPath = stringOption(options, "out", "/content/locomo/results/locomo-results.json");
  const topK = intOption(options, "top-k", 3);
  const userPrefix = stringOption(options, "user-prefix", "locomo");
  const samples = loadSamples(dataPath);
  const store = new MemoryStore(dbPath);
  const records = [];

  try {
    for (const sample of samples) {
      const sampleId = String(sample.sample_id ?? "unknown");
      const userId = `${userPrefix}-${sampleId}`;
      const memories = store.selectMemories(userId, ["semantic", "episodic"], 10000);
      if (memories.length === 0) continue;
      for (const qa of sample.qa ?? []) {
        const evidence = (qa.evidence ?? []).map(String).filter(Boolean);
        if (evidence.length === 0) continue;
        const ranked = rankMemories(String(qa.question ?? ""), memories, topK);
        const selectedIds = ranked.map(locomoDiaId).filter(Boolean);
        records.push({
          sample_id: sampleId,
          category: String(qa.category ?? "unknown"),
          question: String(qa.question ?? ""),
          gold_answer: String(qa.answer ?? ""),
          evidence,
          selected_ids: selectedIds,
          answer_judgment: "not_evaluated",
          score: null,
          hit_at_1: hitAt(evidence, selectedIds, 1),
          hit_at_k: hitAt(evidence, selectedIds, topK)
        });
      }
    }
  } finally {
    store.close();
  }

  const summary = summarize(records, topK);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify({ summary, records }, null, 2), "utf8");
  console.log(JSON.stringify(summary, null, 2));
  console.log(`Wrote ${outPath}`);
  return records.length === 0 ? 1 : 0;
}

async function answerEvaluate(options) {
  const dataPath = stringOption(options, "data", "/content/PSM/benchmark/locomo/data/locomo10.json");
  const dbPath = stringOption(options, "db", "/content/locomo/results/locomo-psm-memory.db");
  const outPath = stringOption(options, "out", "/content/locomo/results/locomo-answer-results.json");
  const topK = intOption(options, "top-k", 50);
  const answerContextK = intOption(options, "answer-context-k", Math.min(5, topK));
  const limit = intOption(options, "limit", 0);
  const checkpointEvery = intOption(options, "checkpoint-every", 10);
  const requestDelayMs = intOption(options, "request-delay-ms", intOption(options, "openrouter-delay-ms", Number(process.env.LOCOMO_REQUEST_DELAY_MS ?? 1500)));
  const requestMaxRetries = intOption(options, "request-max-retries", Number(process.env.LOCOMO_REQUEST_MAX_RETRIES ?? 6));
  const answerModel = stringOption(options, "answer-model", process.env.LOCOMO_ANSWER_MODEL || defaultOpenRouterModel);
  const judgeModel = stringOption(options, "judge-model", process.env.LOCOMO_JUDGE_MODEL || defaultOpenRouterModel);
  const baseUrl = stringOption(options, "base-url", process.env.OPENROUTER_BASE_URL || process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1");
  const apiKey = stringOption(options, "api-key", process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || "");
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is required. Set it in Colab before running answer-evaluate.");

  const existing = loadExistingAnswerResults(outPath);
  const done = new Set(existing.records.map(answerRecordKey));
  const records = [...existing.records];
  const samples = loadSamples(dataPath);
  const store = new MemoryStore(dbPath);
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
        const category = String(qa.category ?? "unknown");
        const key = `${sampleId}\n${category}\n${question}`;
        if (done.has(key)) continue;
        if (limit > 0 && processedThisRun >= limit) {
          writeAnswerResults(outPath, summarizeAnswers(records, topK, answerContextK, answerModel, judgeModel), records);
          return records.length === 0 ? 1 : 0;
        }

        const ranked = rankMemories(question, memories, topK);
        const contextItems = renderRankedContextItems(ranked);
        const retrievedIds = contextItems.flatMap((item) => item.source_ids ?? []);
        const retrievedMemoryIds = contextItems.map((item) => `${item.table}:${item.memory_id ?? item.id ?? ""}`).filter((id) => !id.endsWith(":"));
        const hitAt1 = hitAt(evidence, retrievedIds, 1);
        const hitAtK = hitAt(evidence, retrievedIds, topK);
        const client = { apiKey, baseUrl, answerModel, judgeModel, requestDelayMs, requestMaxRetries };
        const answerContextItems = contextItems.slice(0, answerContextK);
        const answerContextIds = answerContextItems.flatMap((item) => item.source_ids ?? []);
        const answerContextMemoryIds = answerContextItems.map((item) => `${item.table}:${item.memory_id ?? item.id ?? ""}`).filter((id) => !id.endsWith(":"));
        const answerContextHitAtK = hitAt(evidence, answerContextIds, answerContextK);
        const answer = await answerQuestion(client, question, answerContextItems);
        const judgment = await judgeAnswer(client, question, String(qa.answer ?? ""), answer.answer);

        records.push({
          sample_id: sampleId,
          category,
          question,
          gold_answer: String(qa.answer ?? ""),
          evidence,
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
          generated_answer: answer.answer,
          answer_evidence_ids: answer.evidenceIds,
          answer_raw_model_json: answer.raw,
          answer_json_parse_error: answer.parseError,
          judgment: judgment.correct ? "correct" : "incorrect",
          score: judgment.correct ? 1 : 0,
          judge_reasoning: judgment.reasoning,
          answer_model: answerModel,
          judge_model: judgeModel
        });
        done.add(key);
        processedThisRun++;

        if (processedThisRun % checkpointEvery === 0) {
          writeAnswerResults(outPath, summarizeAnswers(records, topK, answerContextK, answerModel, judgeModel), records);
          console.log(`answered=${records.length} this_run=${processedThisRun} accuracy=${answerAccuracy(records).toFixed(4)}`);
        }
      }
    }
  } finally {
    store.close();
  }

  const summary = summarizeAnswers(records, topK, answerContextK, answerModel, judgeModel);
  writeAnswerResults(outPath, summary, records);
  console.log(JSON.stringify(summary, null, 2));
  console.log(`Wrote ${outPath}`);
  return records.length === 0 ? 1 : 0;
}

function report(options) {
  const psmPath = stringOption(options, "psm", "/content/locomo/results/locomo-results.json");
  const baselinesPath = stringOption(options, "baselines", "/content/PSM/benchmark/locomo/baselines/memory-tools.json");
  const outPath = stringOption(options, "out", "/content/locomo/results/locomo-comparison.md");
  const psm = JSON.parse(readFileSync(psmPath, "utf8"));
  const baselines = JSON.parse(readFileSync(baselinesPath, "utf8"));
  const markdown = renderReport(psm, baselines);

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, markdown, "utf8");
  console.log(markdown);
  console.log(`Wrote ${outPath}`);
  return 0;
}

function finish(store, stats, summaryPath = "/content/locomo/results/ingest-summary.json") {
  stats.ended_at = new Date().toISOString();
  const outPath = summaryPath;
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(stats, null, 2), "utf8");
  console.log(JSON.stringify(stats, null, 2));
  return stats.failed > 0 ? 1 : 0;
}

function checkpoint(dbPath, checkpointDir, progressPath, stats, nextIndex) {
  if (checkpointDir) {
    mkdirSync(checkpointDir, { recursive: true });
    copyIfExists(dbPath, join(checkpointDir, basename(dbPath)));
    copyIfExists(`${dbPath}-wal`, join(checkpointDir, `${basename(dbPath)}-wal`));
    copyIfExists(`${dbPath}-shm`, join(checkpointDir, `${basename(dbPath)}-shm`));
  }
  if (progressPath) {
    writeJsonAtomic(progressPath, {
      ...stats,
      next_index: nextIndex,
      checkpointed_at: new Date().toISOString()
    });
  }
}

function loadProgress(path) {
  if (!path || !existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    console.warn(`Ignoring unreadable progress file ${path}: ${error instanceof Error ? error.message : String(error)}`);
    return {};
  }
}

function copyIfExists(from, to) {
  if (existsSync(from)) copyFileSync(from, to);
}

function writeJsonAtomic(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify(value, null, 2), "utf8");
  renameSync(tmp, path);
}

function loadSamples(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function flattenTurns(sample) {
  const conversation = sample.conversation ?? {};
  return Object.keys(conversation)
    .filter((key) => /^session_\d+$/.test(key))
    .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]))
    .flatMap((key) => Array.isArray(conversation[key]) ? conversation[key].map((turn) => ({ ...turn, session: key })) : []);
}

function locomoSourceTimestamp(sample, session) {
  if (!session) return undefined;
  const dateTime = sample.conversation?.[`${session}_date_time`];
  if (typeof dateTime === "string" && dateTime.trim()) return dateTime.trim();
  const sessionNumber = String(session).match(/^session_(\d+)$/)?.[1];
  if (!sessionNumber) return undefined;
  const date = sample.event_summary?.[`events_session_${sessionNumber}`]?.date;
  return typeof date === "string" && date.trim() ? date.trim() : undefined;
}

function locomoDiaId(memory) {
  const tags = parseTags(memory.tags);
  const prefix = "locomo_dia_id:";
  return tags.find((tag) => tag.startsWith(prefix))?.slice(prefix.length) ?? "";
}

function createLocomoIngestRuntime(runtime) {
  return {
    async generateJson(prompt, options) {
      const raw = await runtime.generateJson(prompt, options);
      return hasInvalidLocomoMemoryContent(raw, prompt) ? "invalid locomo memory content" : raw;
    }
  };
}

function hasInvalidLocomoMemoryContent(raw, prompt) {
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
    const memory = parsed.memory;
    const content = typeof memory === "string"
      ? memory.trim()
      : memory && typeof memory === "object" && !Array.isArray(memory) && typeof memory.content === "string"
        ? memory.content.trim()
        : "";
    if (!content) return false;
    if (isLocomoWrapperContent(content)) return true;
    const currentUtterance = extractCurrentUtterance(prompt);
    if (!currentUtterance) return false;
    return !isGroundedInCurrentUtterance(content, currentUtterance);
  } catch {
    return false;
  }
}

function extractCurrentUtterance(prompt) {
  const match = prompt.match(/^Current utterance:\s*"([\s\S]*?)"$/m);
  if (match?.[1]?.trim()) return match[1].trim();

  const unescapedPrompt = prompt.replace(/\\n/g, "\n").replace(/\\"/g, "\"");
  const unescapedMatch = unescapedPrompt.match(/^Current utterance:\s*"([\s\S]*?)"$/m);
  if (unescapedMatch?.[1]?.trim()) return unescapedMatch[1].trim();

  const encodedContent = prompt.match(/"conversation":\[\{"role":"assistant","content":"((?:\\.|[^"\\])*)"\}\]/)?.[1];
  if (!encodedContent) return "";
  try {
    const decodedContent = JSON.parse(`"${encodedContent}"`);
    if (typeof decodedContent !== "string") return "";
    return decodedContent.match(/^Current utterance:\s*"([\s\S]*?)"$/m)?.[1]?.trim() ?? "";
  } catch {
    return "";
  }
}

function isGroundedInCurrentUtterance(content, currentUtterance) {
  const currentTokens = meaningfulTokens(currentUtterance);
  const contentTokens = meaningfulTokens(content);
  if (currentTokens.length === 0 || contentTokens.length === 0) return false;
  const contentSet = new Set(contentTokens);
  const overlap = currentTokens.filter((token) => contentSet.has(token));
  if (overlap.length >= Math.min(2, currentTokens.length)) return true;
  const quoted = content.match(/"([^"]{4,})"/g)?.some((quote) => currentUtterance.includes(quote.slice(1, -1))) ?? false;
  return quoted;
}

function meaningfulTokens(text) {
  const stop = new Set(["the", "and", "but", "you", "your", "for", "with", "that", "this", "what", "have", "been", "said", "asked", "mentioned", "about", "from", "into", "they", "them", "their", "good", "really"]);
  return text.toLowerCase().match(/[a-z0-9+]{3,}/g)?.filter((token) => !stop.has(token)) ?? [];
}

function isLocomoWrapperContent(content) {
  const lower = content.toLowerCase();
  return content.startsWith("{")
    || lower.includes("locomo benchmark conversation turn")
    || lower.includes("normal conversation-memory input")
    || lower.includes("rendered from the benchmark dataset")
    || lower.includes("store only durable memories")
    || lower.includes("current turn to remember:")
    || lower.includes("current utterance:")
    || lower.includes("previous context:")
    || lower.includes("extraction guidance:")
    || lower.includes("do not store")
    || lower.includes("preserve source ids")
    || lower.startsWith("user ")
    || lower.includes(" user ")
    || lower.includes("\"operation\":\"locomo_remember_turn\"")
    || lower.includes("\"operation\": \"locomo_remember_turn\"");
}

function recordRememberResult(stats, source, result) {
  const route = typeof result.route === "string" ? result.route : "";
  const parseError = typeof result.parse_error === "string" ? result.parse_error : "";
  const written = Array.isArray(result.written) ? result.written : [];
  if (parseError) {
    stats.failed++;
    stats.errors.push({ source, error: parseError });
  } else if (route === "ignore" || route === "recall_only") {
    stats.ignored++;
  } else if (written.length > 0) {
    stats.stored++;
  } else {
    stats.ignored++;
  }
}

function renderRankedContextItems(memories) {
  return memories.map((memory) => {
    const table = typeof memory.table === "string" ? memory.table : "memory";
    const sourceIds = unique([...evidenceIdsFromTags(parseTags(memory.tags)), ...evidenceIdsFromSourceId(memory.source_id ?? "")]);
    return {
      id: `${table}:${memory.id}`,
      memory_id: memory.id,
      table,
      content: memory.content,
      score: typeof memory.score === "number" ? memory.score : undefined,
      source_ids: sourceIds,
      source_id: memory.source_id,
      saved_at: memory.created_at,
      reason: "Selected by exact DB retrieval ranking."
    };
  }).filter((item) => typeof item.content === "string" && item.content.trim().length > 0);
}

function parseTags(value) {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function tagValue(tags, key) {
  const prefix = `${key}:`;
  return tags.find((tag) => tag.startsWith(prefix))?.slice(prefix.length) ?? "";
}

function evidenceIdsFromTags(tags) {
  const ids = new Set();
  const diaId = tagValue(tags, "locomo_dia_id");
  if (diaId) ids.add(diaId);
  for (const key of ["related_dia_ids", "locomo_related_dia_ids"]) {
    const value = tagValue(tags, key);
    for (const id of value.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean)) ids.add(id);
  }
  return [...ids];
}

function evidenceIdsFromSourceId(sourceId) {
  const match = String(sourceId).match(/(?:^|:)(D\d+:\d+)$/);
  return match ? [match[1]] : [];
}

function hitAt(evidence, selected, k) {
  return evidence.some((id) => selected.slice(0, k).includes(id));
}

function summarize(records, topK) {
  const denom = records.length || 1;
  return {
    metric: "LOCOMO evidence retrieval only",
    answer_correctness_evaluated: false,
    questions: records.length,
    hit_at_1: records.filter((record) => record.hit_at_1 === true).length / denom,
    [`hit_at_${topK}`]: records.filter((record) => record.hit_at_k === true).length / denom
  };
}

function buildLocomoRememberText({ sample, turns, index, windowSize }) {
  const turn = turns[index];
  const sampleId = String(sample.sample_id ?? "unknown");
  const diaId = String(turn.dia_id ?? "");
  const sourceTimestamp = locomoSourceTimestamp(sample, turn.session);
  const windowStart = Math.max(0, index - windowSize);
  const nearbyTurns = turns.slice(windowStart, index).map((item) => renderTurnLine(item));
  const imageLines = renderImageLines(turn);
  return [
    `Source id: ${sampleId}:${diaId}`,
    `Sample id: ${sampleId}`,
    `Session: ${turn.session || "unknown"}`,
    `Session time: ${sourceTimestamp ?? "unknown"}`,
    `Current speaker: ${turn.speaker ?? "unknown"}`,
    `Current utterance: ${quoteText(turn.text)}`,
    ...imageLines,
    "Previous context:",
    ...(nearbyTurns.length > 0 ? nearbyTurns : ["- none"]),
  ].join("\n");
}

function renderTurnLine(turn) {
  const fields = [
    `${turn.speaker ?? "Unknown"} said: ${quoteText(turn.text)}`,
    turn.query ? `image query: ${turn.query}` : "",
    turn.blip_caption ? `image caption: ${turn.blip_caption}` : ""
  ].filter(Boolean);
  return `- [prior ${turn.session ?? "unknown"} ${turn.dia_id ?? "unknown"}] ${fields.join("; ")}`;
}

function renderImageLines(turn) {
  const lines = [];
  if (turn.query) lines.push(`Image query: ${turn.query}`);
  if (turn.blip_caption) lines.push(`Image caption: ${turn.blip_caption}`);
  if (turn.img_url?.length) lines.push(`Image URLs: ${turn.img_url.join(", ")}`);
  return lines;
}

function quoteText(value) {
  const text = String(value ?? "").trim();
  return text ? `"${text}"` : "\"\"";
}

async function answerQuestion(client, question, contextItems) {
  const context = renderContextForPrompt(contextItems);
  const content = await chatCompletion(client.apiKey, client.baseUrl, client.answerModel, [
    {
      role: "system",
      content: "Answer a LOCOMO benchmark question using only the provided retrieved memories. Return JSON only with exactly this shape: {\"answer\":\"short final answer or I do not know.\",\"evidence_ids\":[\"D1:12\"]}. Do not include reasoning, markdown, citations outside JSON, or extra keys. For when/date questions, return the date or time phrase. For relationship/status questions, return the status. If the memories do not contain the answer, set answer to exactly \"I do not know.\" and evidence_ids to []. evidence_ids must only contain source IDs shown in the retrieved memories."
    },
    {
      role: "user",
      content: `Retrieved memories:\n${context}\n\nQuestion: ${question}\n\nReturn JSON only:`
    }
  ], 180, 0, client.requestDelayMs, client.requestMaxRetries);
  return parseAnswerJson(content);
}

function renderContextForPrompt(contextItems) {
  const context = contextItems.map((item, index) => {
    const table = typeof item.table === "string" ? item.table : "memory";
    const id = typeof item.id === "string" && item.id ? ` id=${item.id}` : "";
    const sources = Array.isArray(item.source_ids) && item.source_ids.length ? ` sources=${item.source_ids.join(",")}` : "";
    const content = typeof item.content === "string" ? item.content : "";
    return `[${index + 1}] [${table}]${id}${sources} ${content}`;
  }).join("\n");
  return context;
}

function parseAnswerJson(value) {
  const trimmed = value.trim();
  const json = trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed;
  try {
    const parsed = JSON.parse(json);
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

function contextPrompt(question, memories) {
  return [
    question,
    "",
    "Select memory context that helps answer this LOCOMO question.",
    "The answerer will only see the context items you return, so include specific names, dates, places, relationships, and facts when relevant.",
    `Candidate memory count: ${memories.length}`
  ].join("\n");
}

function extractContextItems(result) {
  const items = Array.isArray(result.context_items) ? result.context_items : [];
  return items
    .filter((item) => typeof item === "object" && item !== null)
    .map((item) => ({
      id: typeof item.id === "string" ? item.id : undefined,
      table: typeof item.table === "string" ? item.table : "memory",
      content: typeof item.content === "string" ? item.content : "",
      reason: typeof item.reason === "string" ? item.reason : undefined,
      source_ids: sourceIdsFromContextItem(item),
      source_id: typeof item.source_id === "string" ? item.source_id : undefined
    }))
    .filter((item) => item.content.trim().length > 0);
}

function extractSelectedIds(result) {
  const memoryContext = Array.isArray(result.memory_context) ? result.memory_context : [];
  const factContext = Array.isArray(result.fact_context) ? result.fact_context : [];
  return unique([
    ...memoryContext
    .filter((item) => typeof item === "object" && item !== null)
    .map((item) => {
      const tags = Array.isArray(item.metadata?.tags) ? item.metadata.tags.map(String) : [];
      return tagValue(tags, "locomo_dia_id") || evidenceIdsFromSourceId(item.source_id ?? "")[0] || "";
    })
    .filter(Boolean),
    ...factContext
      .filter((item) => typeof item === "object" && item !== null)
      .flatMap((item) => evidenceIdsFromSourceId(item.source_id ?? ""))
  ]);
}

function sourceIdsFromContextItem(item) {
  const ids = new Set();
  if (Array.isArray(item.source_ids)) {
    for (const id of item.source_ids.map(String).filter(Boolean)) ids.add(id);
  }
  if (typeof item.source_id === "string") {
    for (const id of evidenceIdsFromSourceId(item.source_id)) ids.add(id);
  }
  return [...ids];
}

async function judgeAnswer(client, question, goldAnswer, generatedAnswer) {
  const content = await chatCompletion(client.apiKey, client.baseUrl, client.judgeModel, [
    {
      role: "system",
      content: "You are judging a LOCOMO memory benchmark answer. Return JSON only: {\"correct\":true|false,\"reasoning\":\"short reason\"}. Mark correct when the generated answer is semantically consistent with the gold answer. Mark incorrect for missing, contradicted, or unsupported answers."
    },
    {
      role: "user",
      content: `Question: ${question}\nGold answer: ${goldAnswer}\nGenerated answer: ${generatedAnswer}`
    }
  ], 160, 0, client.requestDelayMs, client.requestMaxRetries);
  const parsed = parseJudgeJson(content);
  return {
    correct: parsed.correct,
    reasoning: parsed.reasoning || content.trim()
  };
}

async function chatCompletion(apiKey, baseUrl, model, messages, maxTokens, temperature, requestDelayMs = 1500, requestMaxRetries = 6) {
  let lastError = "";
  for (let attempt = 0; attempt <= requestMaxRetries; attempt++) {
    if (requestDelayMs > 0) await sleep(requestDelayMs);
    const response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        "authorization": `Bearer ${apiKey}`,
        "content-type": "application/json"
      },
      body: JSON.stringify({ model, messages, temperature, max_tokens: maxTokens })
    });
    if (response.ok) {
      const data = await response.json();
      return data.choices?.[0]?.message?.content ?? "";
    }
    const body = await response.text();
    lastError = `Chat completion failed ${response.status}: ${body}`;
    if (response.status !== 429 || attempt >= requestMaxRetries) break;
    const waitMs = rateLimitWaitMs(response, body, attempt);
    console.error(`OpenRouter rate limited; retrying in ${Math.round(waitMs / 1000)}s (attempt ${attempt + 1}/${requestMaxRetries})`);
    await sleep(waitMs);
  }
  throw new Error(lastError);
}

function rateLimitWaitMs(response, body, attempt) {
  const retryAfter = Number(response.headers.get("retry-after"));
  if (Number.isFinite(retryAfter) && retryAfter > 0) return clamp(retryAfter * 1000, 1000, 120000);
  try {
    const parsed = JSON.parse(body);
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseJudgeJson(value) {
  const trimmed = value.trim();
  const json = trimmed.match(/\{[\s\S]*\}/)?.[0] ?? trimmed;
  try {
    const parsed = JSON.parse(json);
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

function cleanAnswer(value) {
  let answer = value.trim();
  answer = answer.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  answer = answer.replace(/^(we need to answer|let'?s answer|analysis|reasoning|thought process)\s*:?.*?\n+/is, "").trim();
  const finalMatch = answer.match(/(?:final answer|answer)\s*:\s*([\s\S]*)/i);
  if (finalMatch?.[1]) answer = finalMatch[1].trim();
  const sentences = answer.split(/(?<=[.!?])\s+/).filter(Boolean);
  if (sentences.length > 3) answer = sentences.slice(-2).join(" ");
  return answer.trim();
}

function summarizeAnswers(records, topK, answerContextK, answerModel, judgeModel) {
  const byCategory = {};
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
    answer_accuracy: answerAccuracy(records),
    hit_at_1: records.filter((record) => record.hit_at_1 === true).length / (records.length || 1),
    [`hit_at_${topK}`]: records.filter((record) => record.hit_at_k === true).length / (records.length || 1),
    answer_context_hit_at_k: records.filter((record) => record.answer_context_hit_at_k === true).length / (records.length || 1),
    top_k: topK,
    answer_context_k: answerContextK,
    answer_model: answerModel,
    judge_model: judgeModel,
    by_category: byCategory
  };
}

function writeAnswerResults(path, summary, records) {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify({ summary, records }, null, 2), "utf8");
  renameSync(tmp, path);
}

function loadExistingAnswerResults(path) {
  if (!existsSync(path)) return { summary: {}, records: [] };
  const parsed = JSON.parse(readFileSync(path, "utf8"));
  return {
    summary: parsed.summary ?? {},
    records: Array.isArray(parsed.records) ? parsed.records : []
  };
}

function answerRecordKey(record) {
  return `${record.sample_id}\n${record.category}\n${record.question}`;
}

function answerAccuracy(records) {
  const denom = records.length || 1;
  return records.reduce((sum, record) => sum + record.score, 0) / denom;
}

function renderReport(psm, baselines) {
  const summary = psm.summary ?? {};
  const answerAccuracy = typeof summary.answer_accuracy === "number" ? summary.answer_accuracy : undefined;
  const topKEntry = Object.entries(summary).find(([key]) => /^hit_at_\d+$/.test(key) && key !== "hit_at_1");
  const topKLabel = topKEntry?.[0]?.replace("hit_at_", "Hit@") ?? "Hit@K";
  const topKValue = typeof topKEntry?.[1] === "number" ? topKEntry[1] : undefined;
  const questions = typeof summary.questions === "number" ? String(summary.questions) : "";
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
    `| PSM Memory | Evidence Hit@1 | ${formatPercent(summary.hit_at_1)} | ${questions} | Retrieved memory contains at least one gold evidence id in the first result. |`,
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

function formatPercent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "";
}

function escapeCell(value) {
  return String(value).replace(/\|/g, "\\|").replace(/\n/g, " ");
}

function buildStoragePrompt(text) {
  return `<|system|>
You are PSM, a memory-management model. Return JSON only.
Choose action: ignore, store_episodic, promote_semantic, update_existing, flag_conflict.
JSON shape: {"action":"store_episodic","memory":{"content":"...","type":"episodic","strength":0.75,"decay_rate":0.02,"emotional_weight":0.2,"confidence":0.8,"tags":[]},"reasoning":"..."}
<|user|>
Remember this conversation turn if useful:
${JSON.stringify(text)}
<|assistant|>
`;
}

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      result[key] = next;
      i++;
    } else {
      result[key] = true;
    }
  }
  return result;
}

function stringOption(options, key, fallback) {
  const value = options[key];
  return typeof value === "string" && value.trim() ? value : fallback;
}

function intOption(options, key, fallback) {
  const parsed = Number(options[key]);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}
