Param()
$ErrorActionPreference = 'Stop'

$py = @'
import yaml
p='config/watchlist.yaml'
d=yaml.safe_load(open(p,encoding='utf-8')) or {}
d.setdefault('notify',{})
d['notify']['mode']='webhook'
d['notify']['generic_webhook_env']='N8N_WEBHOOK_URL'
open(p,'w',encoding='utf-8').write(yaml.safe_dump(d,allow_unicode=True,sort_keys=False))
print('Updated',p)
'@
$tmp = Join-Path $env:TEMP ('switch_notify_' + [guid]::NewGuid().ToString() + '.py')
Set-Content -Encoding UTF8 -Path $tmp -Value $py
& python $tmp
Remove-Item -Force $tmp
