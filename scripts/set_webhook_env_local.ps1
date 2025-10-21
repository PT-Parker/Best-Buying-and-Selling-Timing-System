Param()
$ErrorActionPreference = 'Stop'

$u = $env:N8N_WEBHOOK_URL
if (-not $u -and (Test-Path '.specify\memory\n8n_webhook_url.txt')) { $u = (Get-Content -Raw '.specify\memory\n8n_webhook_url.txt').Trim() }
if (-not $u) {
  $base = $env:N8N_BASE_URL; if (-not $base) { $base = $env:N8N_API_URL; if (-not $base) { $base = 'http://127.0.0.1:5678' } }
  $u = $base.TrimEnd('/') + '/webhook/tw-live-signal'
}
setx N8N_WEBHOOK_URL $u | Out-Null
New-Item -ItemType Directory -Path '.specify\memory' -ea 0 | Out-Null
$u | Out-File -Encoding ascii '.specify\memory\n8n_webhook_url.txt'
Write-Host ('N8N_WEBHOOK_URL=' + $u)

