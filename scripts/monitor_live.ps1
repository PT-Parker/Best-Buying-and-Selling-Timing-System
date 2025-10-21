Param()
$ErrorActionPreference = 'Stop'

pip install -r requirements.txt | Out-Null
$env:PYTHONIOENCODING = 'utf-8'

$py = (Get-Command python).Source
$p = Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'scripts\realtime_monitor.py' -PassThru
Start-Sleep -Seconds 2
if (Test-Path 'state\monitor.pid') {
  Write-Host ('Started monitor. PID=' + (Get-Content -Raw 'state\monitor.pid'))
} else {
  Write-Host ('Started monitor. PID=' + $p.Id)
}

