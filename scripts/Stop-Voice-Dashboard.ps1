$projectRoot = Split-Path -Parent $PSScriptRoot

$config = $null
$configPath = Join-Path $projectRoot 'config.json'
if (Test-Path -LiteralPath $configPath) {
    try { $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json } catch { }
}

# Port lookup, the same as in voice_dashboard.py and Launch-Voice-Dashboard.ps1:
# VOICE_DASHBOARD_PORT, then config.json "dashboard_port", then 8790.
$port = 8790
$configuredPort = 0
if ($config -and [int]::TryParse([string]$config.dashboard_port, [ref]$configuredPort) -and $configuredPort -ge 1 -and $configuredPort -le 65535) {
    $port = $configuredPort
}
if ($env:VOICE_DASHBOARD_PORT) { $port = [int]$env:VOICE_DASHBOARD_PORT }

# Lets the dashboard stop speech recognition, the voice engine, and the push-to-talk helper.
try {
    Invoke-RestMethod -Method Post -ContentType 'application/json' -Body '{}' -Uri "http://127.0.0.1:$port/api/stop-system" -TimeoutSec 30 | Out-Null
} catch { }
# And its own jobs, among them music removal in WSL, which would go on without it. Training is left
# running on purpose; the page can stop it after the next start.
try {
    $state = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/state" -TimeoutSec 30
    foreach ($job in 'restore', 'dataset-cut', 'compare') {
        if ($state.jobs.$job -and $state.jobs.$job.status -eq 'running') {
            Invoke-RestMethod -Method Post -ContentType 'application/json' -Body (@{ key = $job } | ConvertTo-Json) -Uri "http://127.0.0.1:$port/api/stop-job" -TimeoutSec 30 | Out-Null
        }
    }
} catch { }

# Only a process that runs voice_dashboard.py is stopped, so another program on the port is left alone.
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)"
    if ($owner -and $owner.CommandLine -like '*voice_dashboard.py*') {
        # A Restore audio run is a process of its own, which would go on without the dashboard.
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$($owner.ProcessId)" |
            Where-Object { $_.CommandLine -like '*restore_audio.py*' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Stop-Process -Id $owner.ProcessId -Force
    }
}
