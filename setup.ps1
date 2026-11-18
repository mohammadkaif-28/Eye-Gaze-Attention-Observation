$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: python -m pip install uv' }
if (-not (Test-Path .venv/Scripts/python.exe)) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed' }
}
uv pip install --python .venv/Scripts/python.exe -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (-not (Test-Path models/face_landmarker.task)) { & .venv/Scripts/python.exe scripts/download_model.py }
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
Write-Host 'Setup complete. Run .\run.ps1'
