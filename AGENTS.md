# Project working instructions

## GitHub synchronization

The user requests GitHub synchronization after every completed modification to this project.

- Validate the affected code, documentation or generated results before committing. For stage-two visualizations, run scripts/report_stage_two.py, which checks recorded metrics against predictions, and inspect the generated charts.
- If multidimensional scoring code, configs, splits or predictions change, rerun python -m academic_rewriter.assess --neural, scripts/report_multidimensional.py and scripts/report_stage_two.py in that order. The report rejects stale scoring inputs.
- Human-review packets must remain unfilled until actual independent reviewers score them. Keep private blinding keys under ignored .cache; use new paths when exporting another study.
- For academic-refinement data/report changes, run scripts/report_refinement.py and inspect output/academic-refinement/index.html. Preserve the frozen arXivEdits partition and source hashes; use a new versioned study for a changed partition. AI triage and upstream intent signals are not independent human approvals. New paper text and curation packets remain under ignored .cache while public records retain references and provenance. Synthetic pairs are training-only.
- Keep the repository presentation and report text in English.
- Preserve existing user changes. Include related user-requested report changes in the commit; do not silently stage unrelated edits or credentials.
- Refresh SHA256SUMS.txt for changed and newly tracked project files using actual local bytes (including Git LFS checkpoint bytes). Exclude the manifest itself, .git and ignored runtime/cache files.
- Commit the completed change with a descriptive message and push the current branch to its configured upstream. The current origin is https://github.com/Boruy/academic-hybrid-rewriter.git. Check the actual remote and branch before pushing.
- Do not force-push, rewrite published history or change the remote without an explicit request. If remote commits conflict, resolve conservatively and rerun relevant checks.
- If synchronization fails, retain the local commit and clearly report the failure and pending push. Never describe a local commit as synchronized.

This applies when an agent finishes a coherent change; it does not install a background file watcher or create a commit on every intermediate file save.
