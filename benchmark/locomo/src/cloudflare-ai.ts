export const defaultCloudflareModel = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

export interface ChatProviderConfig {
  provider: "cloudflare" | "openrouter";
  apiKey: string;
  baseUrl: string;
  accountId?: string;
  requestDelayMs: number;
  requestMaxRetries: number;
}

export function resolveChatProvider(argv: string[]): ChatProviderConfig {
  const provider = (getOption(argv, "llm-provider", process.env.LOCOMO_LLM_PROVIDER ?? "") || "").toLowerCase();
  const cfToken = process.env.CLOUDFLARE_API_TOKEN || process.env.CF_API_TOKEN || "";
  const cfAccount = process.env.CLOUDFLARE_ACCOUNT_ID || process.env.CF_ACCOUNT_ID || "";
  const orKey = process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || "";
  const resolved = provider === "openrouter"
    ? "openrouter"
    : provider === "cloudflare" || (cfToken && cfAccount)
      ? "cloudflare"
      : orKey
        ? "openrouter"
        : "cloudflare";

  if (resolved === "cloudflare") {
    if (!cfToken || !cfAccount) {
      throw new Error("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID required for Cloudflare Workers AI.");
    }
    const baseUrl = process.env.CLOUDFLARE_AI_BASE_URL
      || `https://api.cloudflare.com/client/v4/accounts/${cfAccount}/ai/v1`;
    return {
      provider: "cloudflare",
      apiKey: cfToken,
      baseUrl,
      accountId: cfAccount,
      requestDelayMs: Number(process.env.LOCOMO_REQUEST_DELAY_MS ?? 400),
      requestMaxRetries: Number(process.env.LOCOMO_REQUEST_MAX_RETRIES ?? 6)
    };
  }

  const apiKey = getOption(argv, "api-key", orKey);
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY required when llm-provider=openrouter.");
  }
  return {
    provider: "openrouter",
    apiKey,
    baseUrl: process.env.OPENROUTER_BASE_URL || process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1",
    requestDelayMs: Number(process.env.LOCOMO_REQUEST_DELAY_MS ?? 1500),
    requestMaxRetries: Number(process.env.LOCOMO_REQUEST_MAX_RETRIES ?? 6)
  };
}

export function defaultAnswerModel(provider: ChatProviderConfig): string {
  if (provider.provider === "cloudflare") {
    return process.env.LOCOMO_ANSWER_MODEL || process.env.CLOUDFLARE_ANSWER_MODEL || defaultCloudflareModel;
  }
  return process.env.LOCOMO_ANSWER_MODEL || "nvidia/nemotron-3-super-120b-a12b:free";
}

export function defaultJudgeModel(provider: ChatProviderConfig): string {
  if (provider.provider === "cloudflare") {
    return process.env.LOCOMO_JUDGE_MODEL || process.env.CLOUDFLARE_JUDGE_MODEL || defaultCloudflareModel;
  }
  return process.env.LOCOMO_JUDGE_MODEL || "nvidia/nemotron-3-super-120b-a12b:free";
}

export async function chatCompletion(
  config: ChatProviderConfig,
  model: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  temperature: number
): Promise<string> {
  let lastError = "";
  for (let attempt = 0; attempt <= config.requestMaxRetries; attempt++) {
    if (config.requestDelayMs > 0) await sleep(config.requestDelayMs);
    const response = await fetch(`${config.baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${config.apiKey}`,
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
    if (response.status !== 429 || attempt >= config.requestMaxRetries) break;
    await sleep(clamp(3000 * 2 ** attempt, 3000, 120000));
  }
  throw new Error(lastError);
}

function getOption(argv: string[], key: string, fallback: string): string {
  const index = argv.indexOf(`--${key}`);
  return index >= 0 && argv[index + 1] && !argv[index + 1].startsWith("--") ? argv[index + 1] : fallback;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
