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

Install the plugin only when embedding PSM Memory in another runtime:

```bash
npm install @psm-memory/pi-plugin
```

npm installs `@psm-memory/sdk` automatically because it is a plugin dependency. A separate SDK install is only needed when using SDK APIs directly.

By default the plugin uses the same shared PSM-owned memory store as the CLI and installed hooks. Pass `dbPath` only for tests or embedded runtimes that need an explicit database.

`enrichPrompt` runs after the user prompts the agent and before the LLM call. It recalls relevant memory and injects it as a private system-context message. Context is produced by the SDK's shared grounded renderer: PSM may turn selected DB rows into concise notes, and SDK validation falls back to complete stored statements if rendering is invalid or ungrounded.

`rememberResponse` queues an asynchronous memory write after the LLM responds or takes a decision. Use `flush()` in tests and `close()` during shutdown.
