Param()
$ErrorActionPreference = 'Stop'

if (-not $env:N8N_WEBHOOK_URL) {
  if (Test-Path '.specify\memory\n8n_webhook_url.txt') { $env:N8N_WEBHOOK_URL = (Get-Content -Raw '.specify\memory\n8n_webhook_url.txt').Trim() }
}
if (-not $env:N8N_WEBHOOK_URL) { Write-Error 'N8N_WEBHOOK_URL not set'; exit 1 }

$body = @{ time=(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); symbol='2330'; action='BUY'; price=1450; reason='smoke' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri $env:N8N_WEBHOOK_URL -ContentType 'application/json' -Body $body | Out-Null
Write-Host ('Sent test payload to ' + $env:N8N_WEBHOOK_URL)

