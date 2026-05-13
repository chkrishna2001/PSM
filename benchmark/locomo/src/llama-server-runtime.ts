import type { GenerateOptions, ModelRuntime } from "psm-sdk";

interface CompletionResponse {
  content?: string;
  choices?: Array<{ text?: string; message?: { content?: string } }>;
}

export class LlamaServerRuntime implements ModelRuntime {
  constructor(private readonly baseUrl: string) {}

  async generateJson(prompt: string, options: GenerateOptions = {}): Promise<string> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/completion`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        prompt,
        n_predict: options.maxTokens ?? 256,
        temperature: options.temperature ?? 0,
        top_k: options.topK ?? 20,
        top_p: options.topP ?? 1,
        stop: ["<|im_end|>", "\n\n<|"]
      })
    });
    if (!response.ok) {
      throw new Error(`llama-server request failed: ${response.status} ${response.statusText} ${await response.text()}`);
    }
    const json = await response.json() as CompletionResponse;
    return json.content ?? json.choices?.[0]?.text ?? json.choices?.[0]?.message?.content ?? "";
  }
}
