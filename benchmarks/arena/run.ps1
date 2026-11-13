[CmdletBinding()]
param(
    [ValidateSet("infinitydb", "mem0", "hybrid")]
    [string[]]$Systems = @("infinitydb", "mem0", "hybrid"),
    [ValidateSet("full", "smoke")]
    [string]$Mode = "full",
    [ValidateRange(0, 100000)]
    [int]$MaxQuestions = 0,
    [ValidateRange(1000, 10000000)]
    [int]$StressIterations = 100000,
    [switch]$SkipLongMemEval,
    [switch]$SkipScopedProbe,
    [switch]$WithReader,
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
function Get-PythonVersion {
    param([string]$Python)
    $raw = & $Python --version 2>&1
    if ($LASTEXITCODE -ne 0 -or $raw -notmatch "Python\s+(\d+)\.(\d+)\.(\d+)") {
        throw "Unable to determine Python version from '$Python'."
    }
    return [version]("$($Matches[1]).$($Matches[2]).$($Matches[3])")
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) { throw "Python was not found on PATH. Install Python 3.10+ and reopen PowerShell." }
$BasePython = $PythonCommand.Source
$PythonVersion = Get-PythonVersion -Python $BasePython
if ($PythonVersion -lt [version]"3.10") { throw "Python 3.10+ is required; found $PythonVersion." }

$Dataset = Join-Path $PSScriptRoot "data\locomo10.json"
if (-not (Test-Path $Dataset)) { throw "LoCoMo dataset not found at '$Dataset'. See benchmarks/arena/README.md for dataset setup." }
$Conversations = @("conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50")
$DatasetHash = (Get-FileHash -Algorithm SHA256 $Dataset).Hash.ToLowerInvariant()
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunManifest = Join-Path $PSScriptRoot "scores\windows-run-$RunStamp.json"
$SelectedSystems = @($Systems | Select-Object -Unique)

Write-Host "InfinityDB Benchmark Arena v2 — Windows" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $PythonVersion ($BasePython)"
Write-Host "Dataset:    $Dataset"
Write-Host "SHA-256:    $DatasetHash"
Write-Host "Mode:       $Mode"
Write-Host "Systems:    $($SelectedSystems -join ', ')"
Write-Host "Stress:     $StressIterations synthetic InfinityDB writes"
Write-Host ""

if ($Mode -eq "smoke") {
    Invoke-Native -File $BasePython -Arguments @("benchmarks/arena/smoke_test.py")
    Write-Host "Smoke checks passed. No full benchmark was run." -ForegroundColor Green
    exit 0
}

Invoke-Native -File $BasePython -Arguments @("-m", "compileall", "-q", "benchmarks/arena")
$Mem0Python = Join-Path $RepoRoot ".arena\venvs\mem0\Scripts\python.exe"
$NeedsMem0Env = ($SelectedSystems -contains "mem0") -or ($SelectedSystems -contains "hybrid")
if ($NeedsMem0Env) {
    if (-not (Test-Path $Mem0Python)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Mem0Python) | Out-Null
        Invoke-Native -File $BasePython -Arguments @("-m", "venv", (Join-Path $RepoRoot ".arena\venvs\mem0"))
    }
    if ($InstallDependencies) {
        Invoke-Native -File $Mem0Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Native -File $Mem0Python -Arguments @("-m", "pip", "install", "-r", "benchmarks/arena/requirements-mem0.txt")
        Invoke-Native -File $Mem0Python -Arguments @("-m", "pip", "install", "-r", "benchmarks/arena/requirements-hybrid.txt")
    } else {
        $Mem0Check = & $Mem0Python -c "import mem0, fastembed, qdrant_client, onnxruntime" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Pinned mem0 dependencies are missing. Re-run with -InstallDependencies. Details: $Mem0Check" }
    }
}

$RunSummaries = @()
foreach ($System in $SelectedSystems) {
    if ($System -eq "infinitydb") { $SystemPython = $BasePython } else { $SystemPython = $Mem0Python }
    foreach ($Conversation in $Conversations) {
        $OutputPath = "benchmarks/arena/results/$System/$Conversation.jsonl"
        $WorkDir = ".arena/$System/$Conversation"
        $WorkerArgs = @("benchmarks/arena/worker.py", "--system", $System, "--conv", $Conversation,
            "--dataset", $Dataset, "--out", $OutputPath, "--workdir", $WorkDir)
        if ($MaxQuestions -gt 0) { $WorkerArgs += @("--max-qa", "$MaxQuestions") }
        if ($Force) { $WorkerArgs += "--force" }
        Write-Host "[$System] $Conversation" -ForegroundColor Yellow
        Invoke-Native -File $SystemPython -Arguments $WorkerArgs
    }
    $RunSummaries += $System
}

$ScoreArgs = @("benchmarks/arena/score.py", "--systems", ($RunSummaries -join ","))
if (-not $WithReader) { $ScoreArgs += "--no-reader" }
else { Write-Host "Reader evaluation requested. Ensure Ollama is running and ARENA_READER_MODEL is pulled." -ForegroundColor Yellow }
Invoke-Native -File $BasePython -Arguments $ScoreArgs

# Separate, explicitly labelled InfinityDB-only scale test. It is not a
# competitor result and must never be merged into the LoCoMo leaderboard.
$StressOutput = "benchmarks/arena/results/stress-$StressIterations.json"
Invoke-Native -File $BasePython -Arguments @("benchmarks/reproducible.py", "--iterations", "$StressIterations", "--output", $StressOutput)

$ScopedOutput = "benchmarks/arena/results/scores/scoped_probe.json"
if (-not $SkipScopedProbe) {
    Invoke-Native -File $BasePython -Arguments @("benchmarks/arena/scoped_probe.py", "--out", $ScopedOutput)
}

$LongMemEvalPath = Join-Path $PSScriptRoot "data\longmemeval_s.json"
$LongMemEvalStatus = "skipped: dataset not present"
if (-not $SkipLongMemEval -and (Test-Path $LongMemEvalPath)) {
    Invoke-Native -File $BasePython -Arguments @("benchmarks/arena/longmemeval_probe.py", "--path", $LongMemEvalPath, "--system", "infinitydb", "--top-k", "5")
    $LongMemEvalStatus = "completed or resumed; inspect results/longmemeval/"
} elseif ($SkipLongMemEval) {
    $LongMemEvalStatus = "skipped by flag"
}

$Manifest = [ordered]@{
    benchmark = "InfinityDB Benchmark Arena v2"
    started_at = $RunStamp
    completed_at = (Get-Date).ToString("o")
    branch = (& git branch --show-current 2>$null)
    commit = (& git rev-parse HEAD 2>$null)
    os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    python = $PythonVersion.ToString()
    python_executable = $BasePython
    dataset = "LoCoMo-10"
    dataset_sha256 = $DatasetHash
    conversations = $Conversations
    systems = $RunSummaries
    max_questions_per_conversation = $MaxQuestions
    reader_enabled = [bool]$WithReader
    force_rerun = [bool]$Force
    stress_iterations = $StressIterations
    stress_result = $StressOutput
    scoped_probe_result = if ($SkipScopedProbe) { "skipped by flag" } else { $ScopedOutput }
    longmemeval = $LongMemEvalStatus
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RunManifest) | Out-Null
$Manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $RunManifest
Write-Host ""
Write-Host "Benchmark completed. Manifest: $RunManifest" -ForegroundColor Green
Write-Host "Raw JSONL rows are preserved under benchmarks/arena/results/."
