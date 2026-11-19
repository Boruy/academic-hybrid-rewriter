# Project working instructions

## Manual code execution

The user wants to run project code personally. Generate or edit code and provide
PowerShell commands; do not execute training, inference, data generators, tests,
report builders or other project scripts unless the user explicitly asks for that
execution. Provide executable validation commands for the user and report them as
pending. Static review, file inspection and authorized GitHub synchronization may continue.

## Project scope and data

- Maintain only the six-layer model: 2 LSTM + 2 Transformer + 2 Mamba blocks, together with its datasets, documentation, training/testing code and necessary setup/configuration.
- Keep repository documentation in English. Maintain the [project overview](docs/Project_Overview_ZH.md) in Chinese at the user's request.
- Preserve frozen dataset bytes, manifests and generator source hashes. Use a new versioned study path for a changed corpus or partition.
- Synthetic pairs are training-only. AI triage and upstream intent signals are not independent human approvals.
- Preserve the frozen arXivEdits partition and source references. New paper text and private blinding keys remain under ignored `.cache`; human-review packets remain unfilled until independent reviewers score them.
- Keep checkpoints, predictions and generated artifacts in ignored `runs/`, `output/` or `.cache/` paths. Preserve existing local training work when cleaning the repository.

## GitHub synchronization

The user requests GitHub synchronization after every completed coherent modification.

- Update the [Chinese project overview](docs/Project_Overview_ZH.md) in the same commit as every completed project change. Revise the affected sections and add a dated changelog entry with the change and its verification status. Base recorded results on the available evidence.
- Validate the affected code or documentation before committing. Respect manual execution: perform static checks and provide pending training/test commands instead of executing project scripts.
- Preserve existing user changes. Stage only related requested changes, with no credentials or ignored runtime files.
- Refresh `SHA256SUMS.txt` for changed and newly tracked project files using actual local bytes. Exclude the manifest itself, `.git` and ignored runtime/cache files.
- Commit with a descriptive message and push the current branch to its configured upstream. Check the actual remote and branch before pushing. The expected origin is `https://github.com/Boruy/academic-hybrid-rewriter.git`.
- Do not force-push, rewrite published history or change the remote without an explicit request. Resolve remote conflicts conservatively and repeat relevant checks, subject to the manual execution rule.
- If synchronization fails, retain the local commit and report the failure and pending push. Never describe a local commit as synchronized.

This workflow does not install a background file watcher or commit every intermediate save.
