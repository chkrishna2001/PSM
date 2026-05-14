# @psm-memory/pi-plugin

Hook helpers for wiring PSM Memory into an agent or chat runtime.

```ts
import { createPsmHooks } from "@psm-memory/pi-plugin";

const psm = createPsmHooks({
  dbPath: "user_memory.db",
  userId: "demo"
});

const prepared = await psm.enrichPrompt({
  prompt: "What should I focus on today?"
});

const llmResponse = await callLlm({
  messages: prepared.messages
});

psm.rememberResponse({
  response: llmResponse
});
```

`enrichPrompt` runs after the user prompts the agent and before the LLM call. It retrieves relevant memory and injects it as a private system-context message.

`rememberResponse` queues an asynchronous memory write after the LLM responds or takes a decision. Use `flush()` in tests and `close()` during shutdown.
