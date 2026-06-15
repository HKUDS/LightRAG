# Run FrameRAG evaluation for ALL ChronoQA stories in parallel.
# Each story gets its own Python process and working directory.
#
# Usage:
#   .\run_all_parallel.ps1                # all 18 stories
#   .\run_all_parallel.ps1 -Stories "1,2,5"  # specific stories
#   .\run_all_parallel.ps1 -MaxLLM 2     # 2 concurrent LLM calls per process (default 1)
#
param(
    [string]$Stories = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
    [int]$MaxLLM = 1,          # keep low to avoid 429; 1 = safest, 2-3 = faster
    [int]$MaxExcerpts = 0,     # 0 = all excerpts; set e.g. 5 for a quick test
    [string]$WorkDir = ".\eval_storage_parallel",
    [string]$ResultsDir = ".\story_results"
)

$storyIds = $Stories -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$null = New-Item -ItemType Directory -Force -Path $ResultsDir

Write-Host "Launching $($storyIds.Count) stories in parallel (MAX_LLM=$MaxLLM)..." -ForegroundColor Cyan

$procs = @()
foreach ($sid in $storyIds) {
    $outFile = Join-Path $ResultsDir "story_$sid.json"
    $env:MAX_LLM      = "$MaxLLM"
    $env:MAX_EXCERPTS = "$MaxExcerpts"
    $logFile = Join-Path $ResultsDir "story_${sid}.log"
    $p = Start-Process python `
        -ArgumentList "run_one_story.py", $sid, $outFile, $WorkDir `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError  "$logFile.err" `
        -PassThru -NoNewWindow
    $procs += [pscustomobject]@{ Process=$p; Story=$sid; Log=$logFile }
    Write-Host "  [story $sid] PID=$($p.Id) -> $outFile"
}

Write-Host "`nWaiting for all stories to finish..." -ForegroundColor Yellow
$done = 0
while ($procs | Where-Object { -not $_.Process.HasExited }) {
    Start-Sleep -Seconds 10
    $finished = $procs | Where-Object { $_.Process.HasExited }
    foreach ($f in $finished) {
        if (-not $f.Reported) {
            $f.Reported = $true
            $exit = $f.Process.ExitCode
            $status = if ($exit -eq 0) { "OK" } else { "FAILED (exit $exit)" }
            Write-Host "  [story $($f.Story)] $status" -ForegroundColor (if ($exit -eq 0) {"Green"} else {"Red"})
            $done++
        }
    }
}

Write-Host "`nAll done. Aggregating results..." -ForegroundColor Cyan
python aggregate_results.py $ResultsDir
