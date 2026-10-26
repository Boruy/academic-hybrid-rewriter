[CmdletBinding()]
param([switch]$Offline)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$VenvRoot = Join-Path $ProjectRoot '.venv'
$ProjectPython = Join-Path $VenvRoot 'Scripts\python.exe'

function Assert-CommandSucceeded([string]$Action) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed. See docs/Environment_Setup.md."
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    # Preserve an existing environment, including its base-package sharing setting.
    if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
        if ($Offline) {
            throw 'Offline mode requires an existing environment with the dependencies installed.'
        }
        & python -m venv $VenvRoot
        Assert-CommandSucceeded 'Virtual environment creation'
    } else {
        Write-Host "Reusing existing environment: $VenvRoot"
    }

    if ($Offline) {
        & $ProjectPython -c "import torch, transformers, tokenizers, numpy, setuptools, wheel"
        Assert-CommandSucceeded 'Offline dependency check'
        $InstallArgs = @('--isolated', 'install', '--no-index', '--no-deps', '--no-build-isolation', '-e', '.')
    } else {
        $InstallArgs = @('install', '--timeout', '120', '--retries', '8', '-e', '.')
    }
    & $ProjectPython -m pip @InstallArgs
    Assert-CommandSucceeded 'Project installation'

    & $ProjectPython -c "import sys, torch, academic_rewriter; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
    Assert-CommandSucceeded 'Environment validation'
} finally {
    Pop-Location
}
