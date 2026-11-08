[CmdletBinding()]
param([ValidateRange(1,1000)][int]$FirstReportGames=16)
$ErrorActionPreference='Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $workspacePath
try {
    if (-not (Test-Path -LiteralPath '.token-hercule-v1')) { throw 'hercule-v1 has no saved credential.' }
    if (Test-Path -LiteralPath '.bot-hercule-v1.lock') {
        $existingBotPid = [int](Get-Content -LiteralPath '.bot-hercule-v1.lock')
        if (Get-Process -Id $existingBotPid -ErrorAction SilentlyContinue) { throw "hercule-v1 is already running (PID $existingBotPid)." }
    }
    $activePath = Join-Path $workspacePath 'runs/hercule-offense-online-active.json'
    $previousLaunch = if (Test-Path -LiteralPath $activePath) { Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json } else { $null }
    if ($previousLaunch -and $previousLaunch.status -eq 'stopped-by-user') { throw 'This continuous run was stopped by the user; start a fresh authorized campaign.' }
    $runLabel = 'hercule-offense-online-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
    $stdoutPath = Join-Path $workspacePath "runs/$runLabel.stdout.log"
    $stderrPath = Join-Path $workspacePath "runs/$runLabel.stderr.log"
    $policyFiles = @('src/hercule.ts','src/hercule-policy.ts','src/hercule-settings.ts','src/hercule-priors.ts','src/scoring-v3.ts','src/planning.ts','src/strategy.ts','src/online.ts','src/online-progress.ts','src/study.ts')
    $sourceHashes = @($policyFiles | ForEach-Object { @{file=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()} })
    Remove-Item Env:BOT_TOKEN,Env:BOT_NONCE,Env:BOT_TRAIN,Env:BOT_TRAIN_MATCHES,Env:BOT_TRAIN_RESUME,Env:BOT_TRAIN_PLAN,Env:BOT_EVAL_MATCHES,Env:BOT_EVAL_MINUTES,Env:TOURNAMENT_ID -ErrorAction SilentlyContinue
    $env:BOT_NAME='hercule-v1'
    $env:BOT_STRATEGY='hercule'
    $env:STDB_HOST='https://maincloud.spacetimedb.com'
    $env:STDB_DB='scrabblebot'
    $nodePath = (Get-Command node.exe).Source
    $botProcess = Start-Process -FilePath $nodePath -ArgumentList @('--import','tsx','src/online.ts') -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $launch = [ordered]@{pid=$botProcess.Id;bot='hercule-v1';policy='hercule-offense-v4';configuration='public-six';defenseWeight=0;status='connecting';startedAt=[DateTime]::UtcNow.ToString('o');firstReportGames=$FirstReportGames;reportIntervalGames=16;continueUntilStopped=$true;lastReportedFullGames=0;runDirectories=@();stdout=$stdoutPath;stderr=$stderrPath;sourceHashes=$sourceHashes}
    if ($previousLaunch) {
        $launch.startedAt=$previousLaunch.startedAt
        $launch.lastReportedFullGames=$previousLaunch.lastReportedFullGames
        $launch.runDirectories=@($previousLaunch.runDirectories)
        $launch.restartAt=[DateTime]::UtcNow.ToString('o')
    }
    $launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $activePath -Encoding utf8
    for ($attempt=0;$attempt -lt 30;$attempt++) {
        if (Test-Path -LiteralPath $stdoutPath) {
            $reportLine = Select-String -LiteralPath $stdoutPath -Pattern '^Reports: (.+)$' | Select-Object -First 1
            if ($reportLine) {
                $launch.runDirectory=$reportLine.Matches[0].Groups[1].Value.Trim()
                $launch.runDirectories=@($launch.runDirectories)+@($launch.runDirectory)
                $launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $activePath -Encoding utf8
                break
            }
        }
        $botProcess.Refresh()
        if ($botProcess.HasExited) { throw "hercule-v1 exited during startup; inspect $stderrPath" }
        Start-Sleep -Milliseconds 500
    }
    $launch | ConvertTo-Json -Depth 8
} finally { Pop-Location }
