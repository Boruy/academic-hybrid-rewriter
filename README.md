# Academic Text Style Enhancement

Transformer + Mamba milestone prototype. Prepared on 24 October 2026 for the Week 9 presentation (25–31 October, per the supplied course requirements).

## Start here

- Submission files: [output/Milestone_Presentation.pptx](output/Milestone_Presentation.pptx), [output/Milestone_Report.pdf](output/Milestone_Report.pdf).
- Chinese speaking script and likely questions: [docs/Presentation_Guide_zh.md](docs/Presentation_Guide_zh.md).
- Real data statistics and model logs: [docs/Progress_Evidence.md](docs/Progress_Evidence.md).
- Data provenance and limitations: [docs/Dataset_Card.md](docs/Dataset_Card.md).
- Architecture and experiment plan: [docs/Architecture_and_Protocol.md](docs/Architecture_and_Protocol.md).
- Before class: [docs/Submission_Checklist_zh.md](docs/Submission_Checklist_zh.md).

## Installation troubleshooting

If pip times out and subsequent commands report missing torch or academic_rewriter, see [environment recovery](docs/Environment_Troubleshooting_zh.md). On this computer the project .venv now uses D:\conda\envs\tee as its base with shared base packages, and the project itself is installed editably into .venv. Use .\.venv\Scripts\python.exe -m ... to select it explicitly. The base environment remains necessary; this setup does not make a standalone dependency copy.

## What actually works

The author-released IteraTeR-HUMAN sentence data have been downloaded with a pinned revision and SHA256 checksums. Cleaning retains 676 arXiv pairs across 95 documents: 491 train, 98 validation and 87 test. Document and exact-text intersections across splits are zero. Train-only tokenization, dynamic padding, target padding masks, tiny BART training, a Mamba reference adapter, three fusion modes, generation and local checkpoint reload are implemented. Both tiny models completed 30 GPU optimization steps on 64 real training pairs. The smoke runs use 16 validation pairs and leave the held-out test set unused for model evaluation.

**This is a development milestone.** The tiny models start from random weights and do not produce useful academic rewrites. The reference Mamba implementation uses a sequential PyTorch scan and has not been checked for numerical parity with official kernels. There is no evidence yet of improved academic tone, meaning preservation or speed. The detector is a transparent heuristic, not a trained classifier. Mamba-only modeling and human style evaluation remain future work.

## Run on this computer

The existing environment is ready. Use .\.venv\Scripts\python.exe explicitly. To register the project offline without recreating the environment:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Offline
~~~

## Initial setup and reproduction

Python 3.10+ is required. The recorded smoke environment used Python 3.11, PyTorch 2.2.0+cu121, Transformers 4.44.2 and an NVIDIA RTX 4060 Laptop GPU. A CPU run is also supported.

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
if ($LASTEXITCODE -ne 0) { throw "Setup failed. Fix the environment before continuing." }
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m academic_rewriter.prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model transformer --output runs/transformer_smoke --steps 30
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model hybrid --backend reference --fusion gated --output runs/hybrid_smoke --steps 30
.\.venv\Scripts\python.exe -m academic_rewriter.demo --checkpoint runs/hybrid_smoke --text "We looked at a lot of samples." --force
~~~

On this computer an existing environment can run the package without installation:

~~~powershell
$env:PYTHONPATH = "$PWD\src"
& D:\conda\envs\tee\python.exe -m academic_rewriter.prepare
& D:\conda\envs\tee\python.exe -m academic_rewriter.run --tiny --model hybrid --output runs/hybrid_smoke
~~~

The demo deliberately displays the smoke-checkpoint warning. A model-generated sentence is a suggestion requiring author review. Number checks alone cannot validate meaning.

## Next experiment with pretrained BART

~~~powershell
.\.venv\Scripts\python.exe -m academic_rewriter.run --model transformer --pretrained facebook/bart-base --output runs/bart_baseline --steps 300 --train-limit 0 --val-limit 0 --lr 0.00005 --batch-size 2
.\.venv\Scripts\python.exe -m academic_rewriter.run --model hybrid --pretrained facebook/bart-base --backend reference --freeze-backbone --output runs/bart_adapter --steps 300 --train-limit 0 --val-limit 0 --lr 0.0001 --batch-size 2
~~~

These commands are **next-stage starts**, not completed experiments or an early-stopping trainer. The current runner uses a fixed step count. Add epoch-level validation, checkpoint selection, scheduler, mixed precision and gradient accumulation before the main study. Keep token limits and decoding settings matched. Full fine-tuning at 128 tokens on this GPU has not been profiled.

Official Mamba is an explicit optional backend. Set up a separate Linux/WSL2 environment, follow the current upstream installation instructions, and validate forward/backward and checkpoint behavior before selecting --backend official. No automatic backend substitution occurs. See https://github.com/state-spaces/mamba .

## Complete repository and Git LFS

This repository includes source code, tests, setup scripts, original and processed data, all five local model checkpoints with their tokenizers and backbone configurations, training logs, English and Chinese documentation, original proposal/slides, final deliverables and the existing material ZIP archives. Virtual environments, caches, bytecode and generated package metadata are excluded; install dependencies with the setup instructions above. The existing ZIP files are earlier snapshots; the working tree contains the current complete project.

Model weights use Git LFS, including the two BART checkpoints larger than GitHub's regular file limit. Install Git LFS before cloning, then download and verify the actual weights:

~~~powershell
git lfs install
git clone https://github.com/Boruy/academic-hybrid-rewriter.git
cd academic-hybrid-rewriter
git lfs pull
git lfs fsck
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
~~~

Use a Git clone with Git LFS for the complete project. GitHub source ZIP downloads may contain LFS pointers instead of model bytes. [Full contents and publication notes](docs/GitHub_Publishing_zh.md) explain the scope and verification. [SHA256SUMS.txt](SHA256SUMS.txt) records the included project files.

## Provenance

IteraTeR: https://github.com/vipulraheja/iterater and https://huggingface.co/datasets/wanyu/IteraTeR_human_sent . The dataset card lists Apache-2.0. Preserve upstream attribution and review any source-text terms before redistribution. Source code here is newly authored for this coursework; no license for the team's original code is chosen on the team's behalf.

The course syllabus itself is not in this folder. Deliverable limits, weighting and dates follow the requirements supplied in the conversation. Team spelling follows Proposal.pdf (Licheng Wang), while MLPre1.pptx spells the given name Lichen. Verify the roster before submission.
