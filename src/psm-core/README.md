# psm-sdk

TypeScript SDK for PSM memory routing, SQLite storage, recall planning, ranking, and local GGUF runtimes.

```ts
import { HeuristicRuntime, MemoryStore, PsmService } from "psm-sdk";
```

`NodeLlamaRuntime` uses `node-llama-cpp` when a GGUF model path is provided.
