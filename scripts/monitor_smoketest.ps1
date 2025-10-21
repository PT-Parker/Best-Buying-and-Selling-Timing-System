Param()
$ErrorActionPreference = 'Stop'

pip install -r requirements.txt | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
python scripts\realtime_monitor.py --force --max-seconds 120
Write-Host 'Monitor smoketest finished (stopped after 120s)'

