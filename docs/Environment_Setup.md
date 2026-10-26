# Environment setup

## New installation

Use Python 3.10 or later. On Windows, scripts/setup.ps1 creates .venv only if its interpreter does not already exist, then installs the project and checks imports. Installation failure stops the script.

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
if ($LASTEXITCODE -ne 0) { throw 'Installation failed. Resolve the dependency error before continuing.' }
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

For download timeouts, the setup script uses a 120-second timeout and eight retries. Use the selected environment's python -m pip rather than a pip executable belonging to another environment.

## Original computer

The working .venv shares installed packages from D:\conda\envs\tee with include-system-site-packages=true. That base environment must remain present. Copying this virtual environment does not provide a portable installation.

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Offline
.\.venv\Scripts\python.exe -c "import torch, academic_rewriter; print(torch.__version__); print(torch.cuda.is_available())"
~~~

Offline setup requires dependencies, setuptools and wheel to be installed already. It registers this project without downloading packages. Do not recreate .venv over the working shared environment, because doing so can disable access to its base packages.

The local .cache/huggingface directory stores pretrained dependencies used by --local-files-only. Keep it for offline training. Neither the cache nor the virtual environment belongs in the Git repository. Saved experiment checkpoints include their own tokenizer and backbone configuration for local evaluation.

The original smoke environment used Python 3.11, PyTorch 2.2.0+cu121, Transformers 4.44.2, Tokenizers 0.19.1 and NumPy 1.26.4. The optional official Mamba backend requires a separately validated installation. No automatic backend substitution occurs.
