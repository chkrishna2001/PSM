# @psm-memory/sdk

TypeScript SDK for PSM Memory routing, SQLite storage, grounded recall, shared config, embeddings, and local GGUF runtimes.

```ts
import { MemoryStore, NodeLlamaRuntime, PsmService } from "@psm-memory/sdk";
```

`NodeLlamaRuntime` uses `node-llama-cpp` when a GGUF model path is provided.

SQLite storage is isolated behind an internal adapter. The default adapter uses Node's built-in `node:sqlite`, so the default SDK install does not depend on a native npm SQLite package. See `docs/sqlite-storage-adapter.md` in the repository for the driver boundary and tradeoffs.

Core guarantees:

- PSM may plan retrieval and render agent context, but rendered context is validated against selected DB rows before injection.
- If context rendering is invalid or ungrounded, the SDK falls back to complete stored statements instead of hard-truncating raw memories.
- Shared config resolves the per-user memory directory, default user id, embedding settings, runtime settings, and daemon behavior.
- `TraceModelRuntime` can write opt-in local JSONL traces with full PSM prompts and raw outputs for debugging and feedback collection.
- User content stays in SQLite; model weights do not store private memory.
