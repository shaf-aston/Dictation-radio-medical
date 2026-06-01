#!/bin/bash
# run= bash ./scripts/run_web.sh

PORT=8005

powershell.exe -NoProfile -Command "\$pids = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; if (\$pids) { Stop-Process -Id \$pids -Force }" >/dev/null 2>&1

python -m src.ui.web_app
