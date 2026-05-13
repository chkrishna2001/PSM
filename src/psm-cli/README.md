# @psm-memory/cli

Command line interface for PSM Memory operations.

```bash
psm init --db user_memory.db
psm remember --db user_memory.db --user demo --llm-response "User prefers concise answers."
psm recall --db user_memory.db --user demo --question "How should I answer?"
```

Pass `--model path/to/psm.gguf` to use the local GGUF runtime.
