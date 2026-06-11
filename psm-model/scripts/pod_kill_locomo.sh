#!/bin/bash
pkill -f 'ingest-psm-model.js' 2>/dev/null || true
pkill -f 'psm_model.remember_cli' 2>/dev/null || true
sleep 1
pgrep -af 'ingest-psm|remember_cli' || echo "locomo procs stopped"
