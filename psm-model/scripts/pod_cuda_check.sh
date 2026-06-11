#!/bin/bash
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
pgrep -af 'python.*remember|ingest-psm' | head -5 || echo "no python ingest"
/usr/bin/python3 -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())" 2>&1
