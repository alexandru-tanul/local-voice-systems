$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

$config = $null
$configPath = Join-Path $projectRoot 'config.json'
if (Test-Path -LiteralPath $configPath) {
    try {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    } catch {
        Write-Warning 'config.json could not be parsed; continuing with the default settings.'
    }
}

# Port lookup, the same as in voice_dashboard.py and Stop-Voice-Dashboard.ps1:
# VOICE_DASHBOARD_PORT, then config.json "dashboard_port", then 8790.
$port = 8790
$configuredPort = 0
if ($config -and [int]::TryParse([string]$config.dashboard_port, [ref]$configuredPort) -and $configuredPort -ge 1 -and $configuredPort -le 65535) {
    $port = $configuredPort
}
if ($env:VOICE_DASHBOARD_PORT) { $port = [int]$env:VOICE_DASHBOARD_PORT }
$dashboardUrl = "http://127.0.0.1:$port"
# The browser keeps the page's saved settings under this address, so it stays localhost.
$pageUrl = "http://localhost:$port"

try {
    $existing = Invoke-WebRequest -UseBasicParsing -Uri "$dashboardUrl/api/system-state" -TimeoutSec 2
    if ($existing.StatusCode -eq 200) {
        Start-Process $pageUrl
        Write-Host 'The voice dashboard is already running.'
        exit 0
    }
} catch {
    # Expected when the dashboard is not running yet.
}

# Python lookup order: config.json "dashboard_python", relay_env, a local .venv, then python or py on PATH.
$pythonCandidates = @()
if ($config -and $config.dashboard_python) { $pythonCandidates += [string]$config.dashboard_python }
$pythonCandidates += (Join-Path $projectRoot 'relay_env\Scripts\python.exe')
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
    throw 'Python was not found. Install 64-bit Python 3.12, or set "dashboard_python" in config.json.'
}

& $python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 or newer is required. Run Setup-Relay-Env.bat first.' }

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
            Start-Process $pageUrl
            Write-Host "Dashboard ready at $pageUrl"
            exit 0
        }
    } catch {
        Start-Sleep -Milliseconds 300
    }
}

throw 'Dashboard did not become ready within 30 seconds. Check voice_dashboard.err.log.'
