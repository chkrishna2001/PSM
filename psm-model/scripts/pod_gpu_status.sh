#!/bin/bash
echo "=== $(date -u +%H:%M:%SZ) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
ps aux 2>/dev/null | grep -E 'remember_cli|eval_checkpoint|psm_model|ingest-psm' | grep -v grep | head -5 || echo "no psm python procs"
ps aux 2>/dev/null | grep -E 'node.*ingest' | grep -v grep | head -3 || echo "no node ingest"
