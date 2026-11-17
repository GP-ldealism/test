$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\Study\Postgraduate\人工智能项目'
$env:CUDA_CACHE_PATH = Join-Path (Get-Location) '.cache/cuda'
$env:PYTHONUTF8 = '1'
& 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe' -u .cache/formal_followup_20261116.py
exit $LASTEXITCODE
