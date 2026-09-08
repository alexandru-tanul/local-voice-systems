$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$dashboardUrl = 'http://127.0.0.1:8790'

try {
    $existing = Invoke-WebRequest -UseBasicParsing -Uri "$dashboardUrl/api/system-state" -TimeoutSec 2
    if ($existing.StatusCode -eq 200) {
        Start-Process 'http://localhost:8790'
        Write-Host 'The voice dashboard is already running.'
        exit 0
    }
} catch {
    # Expected when the dashboard is not running yet.
}

# Python lookup order: config.json "dashboard_python", a local .venv, then python or py on PATH.
$pythonCandidates = @()
$configPath = Join-Path $projectRoot 'config.json'
if (Test-Path -LiteralPath $configPath) {
    try {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if ($config.dashboard_python) { $pythonCandidates += [string]$config.dashboard_python }
    } catch {
        Write-Warning 'config.json could not be parsed; continuing with the default Python lookup.'
    }
}
$pythonCandidates += (Join-Path $projectRoot '.venv\Scripts\python.exe')
$python = $pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $python) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $python = $pythonCommand.Source }
}
if (-not $python) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) { $python = $pyLauncher.Source }
}
if (-not $python) {
    throw 'Python was not found. Install Python 3.11 or newer, or set "dashboard_python" in config.json.'
}

$stdoutLog = Join-Path $projectRoot 'voice_dashboard.out.log'
$stderrLog = Join-Path $projectRoot 'voice_dashboard.err.log'
$process = Start-Process -FilePath $python `
    -ArgumentList 'voice_dashboard.py' `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host 'Starting the voice dashboard...'
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        $detail = if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Tail 12 | Out-String } else { '' }
        throw "Dashboard exited during startup.`n$detail"
    }
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$dashboardUrl/api/system-state" -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Start-Process 'http://localhost:8790'
            Write-Host 'Dashboard ready at http://localhost:8790'
            exit 0
        }
    } catch {
        Start-Sleep -Milliseconds 300
    }
}

throw 'Dashboard did not become ready within 30 seconds. Check voice_dashboard.err.log.'
