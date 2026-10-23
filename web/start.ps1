param([ValidateRange(1024, 65535)][int]$Port = 7860)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw "Python environment not found: $taskPython. See web/README.md."
}
& $taskPython (Join-Path $PSScriptRoot 'app.py') --port $Port
exit $LASTEXITCODE
