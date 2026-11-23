# Environment setup

## New installation

Use Python 3.10 or later. On Windows, `scripts/setup.ps1` reuses an existing `.venv` interpreter or creates the environment, then installs the project and checks imports. Each block uses `P:\MLProject` as the checkout path. If your checkout is elsewhere, adjust the first `Set-Location` in the block.

~~~powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'scripts/setup.ps1'
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed. See the output above.' }
    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed. See the output above.' }
}
~~~

The setup script uses a 120-second download timeout and eight retries. The optional evaluation package can be installed after setup to add SacreBLEU.

~~~powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    & '.\.venv\Scripts\python.exe' -m pip install --timeout 120 --retries 8 -e '.[evaluation]'
    if ($LASTEXITCODE -ne 0) { throw 'Optional evaluation dependencies could not be installed. See the output above.' }
}
~~~

## Original computer

The working .venv shares installed packages from D:\conda\envs\tee with include-system-site-packages=true. That base environment must remain present. Copying this virtual environment does not provide a portable installation.

~~~powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'scripts/setup.ps1' -Offline
    if ($LASTEXITCODE -ne 0) { throw 'Offline setup failed. See the output above.' }
    & '.\.venv\Scripts\python.exe' -c "import torch, academic_rewriter; print(torch.__version__); print(torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed. See the output above.' }
}
~~~

Offline setup requires dependencies, setuptools and wheel to be installed already. It registers this project without downloading packages. Do not recreate .venv over the working shared environment, because doing so can disable access to its base packages.

The six-layer model initializes from scratch and trains its tokenizer on training text only. It does not need pretrained model downloads. Saved checkpoints include their own tokenizer and six-layer configuration for local evaluation. Existing caches are preserved locally; neither caches nor the virtual environment belong in the Git repository. Checkpoints and predictions are ignored under runs/ or .cache/.

The original environment used Python 3.11, PyTorch 2.2.0+cu121, Transformers 4.44.2, Tokenizers 0.19.1 and NumPy 1.26.4. The six-layer model uses the included PyTorch Mamba reference recurrence and does not expose an official Mamba backend.
