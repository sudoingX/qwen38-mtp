param(
    [string]$Model = 'G:\.lmstudio\models\hub\models--empero-ai--Qwen3.8-27B-Ridge-GGUF\blobs\95580dbdaad579582ee898257116abc18d7f3625a00c16a15735d41444a09f5e',
    [int]$Port = 8080,
    [int]$Context = 65536,
    [int]$GpuLayers = 999,
    [int]$DraftMax = 2,
    [string]$Server = "$PSScriptRoot\tools\llama.cpp\llama-server.exe"
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Server)) { throw "llama-server not found: $Server" }
if (-not (Test-Path -LiteralPath $Model)) { throw "Model not found: $Model" }

& $Server -m $Model -c $Context -ngl $GpuLayers -fa 1 `
  --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp `
  --spec-draft-n-max $DraftMax --parallel 1 `
  --host 127.0.0.1 --port $Port
