# @psm-memory/pi-plugin

Hook helpers for wiring shared local PSM Memory into an agent or chat runtime.

```ts
import { createPsmHooks } from "@psm-memory/pi-plugin";

const psm = createPsmHooks({
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

By default the plugin uses the same shared PSM-owned memory store as the CLI and installed hooks. Pass `dbPath` only for tests or embedded runtimes that need an explicit database.

`enrichPrompt` runs after the user prompts the agent and before the LLM call. It recalls relevant memory and injects it as a private system-context message. Injected memory is copied from exact stored DB rows rather than generated as free-form memory text.

`rememberResponse` queues an asynchronous memory write after the LLM responds or takes a decision. Use `flush()` in tests and `close()` during shutdown.
