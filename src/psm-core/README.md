# @psm-memory/sdk

TypeScript SDK for PSM Memory routing, SQLite storage, grounded recall, shared config, embeddings, and local GGUF runtimes.

```ts
import { MemoryStore, NodeLlamaRuntime, PsmService } from "@psm-memory/sdk";
```

`NodeLlamaRuntime` uses `node-llama-cpp` when a GGUF model path is provided.

Core guarantees:

- PSM may plan and rank retrieval, but agent-injected context is copied from exact stored DB rows.
- Shared config resolves the per-user memory directory, default user id, embedding settings, runtime settings, and daemon behavior.
- User content stays in SQLite; model weights do not store private memory.
