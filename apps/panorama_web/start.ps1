param([int]$Port = 8010)
$ErrorActionPreference = 'Stop'
$PiersAppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PiersLocalHome = if ($env:PIERS_WEB_HOME) { $env:PIERS_WEB_HOME } else { Join-Path $env:LOCALAPPDATA 'PIERSWorkbench' }
$PiersEnvDir = Join-Path $PiersLocalHome 'venv'
$PiersPython = Join-Path $PiersEnvDir 'Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $PiersLocalHome | Out-Null
if (-not (Test-Path -LiteralPath $PiersPython)) {
    python -m venv $PiersEnvDir
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
    & $PiersPython -m pip install -r (Join-Path $PiersAppDir 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
$PiersLog = Join-Path $PiersLocalHome 'web.stdout.log'
$PiersErr = Join-Path $PiersLocalHome 'web.stderr.log'
$PiersAppFile = Join-Path $PiersAppDir 'app.py'
try { $PiersHealth = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2 } catch { $PiersHealth = $null }
if (-not $PiersHealth) {
    Start-Process -FilePath $PiersPython -ArgumentList @('-X', 'utf8', ('"' + $PiersAppFile + '"'), '--port', "$Port") -WorkingDirectory $PiersAppDir -WindowStyle Hidden -RedirectStandardOutput $PiersLog -RedirectStandardError $PiersErr
}
Write-Host "PIERS: http://127.0.0.1:$Port"
