[CmdletBinding()]
param(
    [ValidateSet("infinitydb", "mem0", "hybrid")]
    [string[]]$Systems = @("infinitydb", "mem0", "hybrid"),
    [string]$DatasetPath = "benchmarks/arena/data/longmemeval_s_cleaned.json",
    [ValidateRange(0, 500)]
    [int]$Limit = 0,
    [ValidateRange(1, 50)]
    [int]$TopK = 5,
    [switch]$DownloadDataset,
    [switch]$InstallDependencies,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Invoke-Native {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1} {2}" -f $LASTEXITCODE, $File, ($Arguments -join " "))
    }
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) { throw "Python was not found on PATH. Install Python 3.10+." }
$Python = $PythonCommand.Source
$DatasetFullPath = [System.IO.Path]::GetFullPath($DatasetPath)
if (-not (Test-Path $DatasetFullPath)) {
    if (-not $DownloadDataset) {
        throw "LongMemEval dataset missing. Re-run with -DownloadDataset or pass -DatasetPath."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DatasetFullPath) | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $DatasetUrl = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    Write-Host "Downloading MIT-licensed LongMemEval-S cleaned dataset (~265 MB)..."
    Invoke-WebRequest -Uri $DatasetUrl -OutFile $DatasetFullPath -UseBasicParsing
}

$Mem0Python = Join-Path $RepoRoot ".arena\venvs\mem0\Scripts\python.exe"
$NeedsEmbeddingEnv = ($Systems -contains "mem0") -or ($Systems -contains "hybrid")
if ($NeedsEmbeddingEnv) {
    if (-not (Test-Path $Mem0Python)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Mem0Python) | Out-Null
        Invoke-Native -File $Python -Arguments @("-m", "venv", (Join-Path $RepoRoot ".arena\venvs\mem0"))
    }
    if ($InstallDependencies) {
        Invoke-Native -File $Mem0Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Native -File $Mem0Python -Arguments @("-m", "pip", "install", "-r", "benchmarks/arena/requirements-mem0.txt")
        Invoke-Native -File $Mem0Python -Arguments @("-m", "pip", "install", "-r", "benchmarks/arena/requirements-hybrid.txt")
    } else {
        $Check = & $Mem0Python -c "import mem0, fastembed, qdrant_client, onnxruntime, numpy, ollama" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Embedding dependencies missing. Re-run with -InstallDependencies. Details: $Check" }
    }
}

Invoke-Native -File $Python -Arguments @("-m", "compileall", "-q", "benchmarks/arena")
$ArenaArgs = @(
    "benchmarks/arena/longmemeval_arena.py",
    "--path", $DatasetFullPath,
    "--systems", ($Systems -join ","),
    "--top-k", "$TopK"
)
if ($Limit -gt 0) { $ArenaArgs += @("--limit", "$Limit") }
if ($Force) { $ArenaArgs += "--force" }
$ArenaPython = $Python
if ($NeedsEmbeddingEnv) { $ArenaPython = $Mem0Python }
Invoke-Native -File $ArenaPython -Arguments $ArenaArgs
Write-Host "LongMemEval comparison complete. Results are under benchmarks/arena/results/longmemeval/." -ForegroundColor Green
