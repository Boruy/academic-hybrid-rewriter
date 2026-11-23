# Academic revision datasets

The [word formality corpus](../data/word-formality-v1/README.md) supervises the word module during the same run as sentence-pair training. Its 76 synthetic examples are training only. Use separate, independently reviewed real word pairs for the optional word test.

All original dataset files, frozen splits, source hashes and curation records are retained. Dataset code is in `academic_rewriter.prepare` and `academic_rewriter.synthetic`. A changed study must use a new versioned path rather than replace frozen data.

Run the PowerShell blocks with the project environment installed. Each block sets the checkout path to `P:\MLProject`. Adjust that first path if your checkout is elsewhere. [Environment setup](Environment_Setup.md)

## Active paired corpus

`data/academic-pairs-v1` contains:

| Split | Real pairs | Synthetic pairs | Total |
| --- | ---: | ---: | ---: |
| Training | 491 | 12,000 | 12,491 |
| Validation | 98 | 0 | 98 |
| Test | 87 | 0 | 87 |

The 12,000 synthetic pairs use 48 authored semantic frames across 24 technical domains. Source and target share semantic slots and describe hypothetical results. Records include provenance, hashes, protected terms and explicit synthetic/unverified training-only flags.

This is template-generated weak supervision. Shared slots and automatic guards do not establish independent semantic equivalence or academic quality. Variations in quantities and terms are not independent naturally occurring writing examples. The 491 real revisions are mixed-intent supervision rather than uniformly academic-style positives.

The corpus manifest records composition, the exact generator hash and both original and prepared split hashes. Validation and test files are byte-for-byte copies of `data/processed`. See the [corpus README](../data/academic-pairs-v1/README.md).

## Original IteraTeR data

Source: Wanyu Du et al., Understanding Iterative Revision from Human-Written Text (ACL 2022). The pinned `wanyu/IteraTeR_human_sent` release is revision `e22e0371dac444239b944f9293f5b491d62b73f0`. Source bytes and hashes are retained in `data/raw/` and `data/raw/manifest.json`; attribution and the upstream license are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and `data/IteraTeR_LICENSE.txt`.

The downloaded human sentence release has 3,254 train, 400 dev and 364 test records (4,018 total). Processing selects arXiv documents and style, clarity or fluency intents; normalizes Unicode and whitespace; removes identities and duplicates; requires 5-100 words per side and a target/source ratio of 0.5-2.0; and rejects changed numeric multisets.

The retained 676 pairs are split by connected document groups with seed 42. Groups connect revisions of a document and documents sharing an exact normalized source or target. This differs from the upstream sentence splits.

| Split | Pairs | Documents | Clarity | Fluency | Style |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 491 | 76 | 355 | 123 | 13 |
| Validation | 98 | 9 | 69 | 26 | 3 |
| Test | 87 | 10 | 67 | 12 | 8 |

Recorded cross-split document and exact-text overlaps are zero. This does not exclude semantic near duplicates or paper-family overlap. Intent labels and unchanged numbers do not certify academic tone or factual equivalence. The already inspected validation/test sets are diagnostics, not unseen confirmatory evidence.

## Preserved curation records

`data/curation/iterater_audit_100.jsonl` retains the preliminary audit. `data/curation/arxivedits/` retains source references, hashes, the frozen partition, AI triage and a summary. The arXivEdits source revision is `113913d68522c1c8428d8e191d5c9bed633da6ce`.

These records do not add new paper text to the public repository. Original paper text and private reviewer mappings remain in ignored `.cache` locations. Paper-specific licenses still apply. The public records report zero independently human-verified new pairs; upstream intent signals and AI triage are not independent human approvals. The historical report and curation interfaces have been removed, but the frozen references and provenance remain intact.

## Reviewed training expansion

The first expansion uses the **118** screened candidates from the frozen arXivEdits training partition. Preliminary AI triage shortlisted 19 of them for human review. The two local training packets currently have zero recorded human decisions.

Each independent reviewer opens their own packet in `.cache/academic-refinement/review-v2/train/`, `curator_1.html` or `curator_2.html`. Review whether the source needs academic refinement and whether the edit improves the expression while retaining the facts, methods, quantities, qualifiers and scope. Both reviewers must select Academic Style, score meaning preservation and target fluency at least 4/5, mark the source as needing revision and the target as suitable, and report zero major factual errors.

Use **Download my reviews** in each packet. Place the exports `train_curator_1.jsonl` and `train_curator_2.jsonl` in the same local review directory. Partial exports are supported, and a pair enters training when both reviews are complete and pass the criteria.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $reviews = '.cache/academic-refinement/review-v2/train'
    $firstReview = "$reviews/train_curator_1.jsonl"
    $secondReview = "$reviews/train_curator_2.jsonl"
    foreach ($review in @($firstReview, $secondReview)) {
        if (-not (Test-Path -LiteralPath $review -PathType Leaf -ErrorAction Stop)) {
            throw "Place the completed curator export at $review."
        }
    }
    & $python scripts/expand_academic_pairs.py --version 2 --reviews $firstReview $secondReview
    if ($LASTEXITCODE -ne 0) { throw 'v2 dataset build failed. See the output above.' }
}
```

The builder binds reviewed text to the frozen references, checks for duplicate pairs and overlap with protected documents and held-out text, and writes `.cache/academic-pairs-v2` once it has at least one jointly approved pair. It appends approved real pairs to the v1 training bytes and copies the original 98 validation and 87 test pairs byte for byte. The local manifest records review, source and dataset fingerprints and counts. The local experiment config reuses the v8 recipe and assigns a dedicated run directory. Choose an unused version number for a later expansion of the review pool. [Complete v2 training and validation commands](LexiFAR_Training.md#train-on-the-reviewed-expansion)

## Author revision expansion

The v3 experiment adds actual author revisions from the pinned [arXivEdits training alignment file](https://github.com/chaojiang06/arXivEdits/blob/113913d68522c1c8428d8e191d5c9bed633da6ce/data/sentence_alignment/train.json). The cached input is `.cache/arxivedits/train_alignment.json`. Its SHA-256 must match the entry in `data/curation/arxivedits/frozen_partition.json`.

The builder takes one-to-one sentence alignments between successive paper versions, with at least 75% word-order similarity. It checks length, numbers, selected negation and qualifier words, acronyms, equation and citation placeholders, and the two version license records. It includes qualifying punctuation and case changes, excludes duplicate pairs and protected papers or text, and records the source sentence IDs and hashes. New rows are training-only and carry `human_verified: false`. Their quality supervision comes from the authors' revisions and the automated screen.

A read-only inventory of the pinned input counted 9,786 screened revisions from 434 papers, including 8,331 word changes and 1,455 punctuation or case changes. The v1 real pairs bring the count to 10,277 before v8 encoding checks. The completed local v3 manifest records 10,264 real and 12,000 synthetic training pairs. The block below downloads the pinned alignment when the file is absent, verifies its SHA-256 and builds the dataset if needed. It prints the counts from the local manifest.

```powershell
& {
    Set-Location -LiteralPath 'P:\MLProject' -ErrorAction Stop
    $python = '.\.venv\Scripts\python.exe'
    $cache = '.cache/arxivedits'
    $alignment = "$cache/train_alignment.json"
    $alignmentUrl = 'https://raw.githubusercontent.com/chaojiang06/arXivEdits/113913d68522c1c8428d8e191d5c9bed633da6ce/data/sentence_alignment/train.json'
    $expectedHash = '6ee44429681edb7eab39a90d9272aa28e79473458d8d5b94cb99394c173ad184'
    $data = '.cache/academic-pairs-v3'

    if (-not (Test-Path -LiteralPath $alignment -PathType Leaf -ErrorAction Stop)) {
        New-Item -Path $cache -ItemType Directory -Force -ErrorAction Stop | Out-Null
        $download = "$alignment.download"
        Invoke-WebRequest -Uri $alignmentUrl -UseBasicParsing -OutFile $download -TimeoutSec 180 -ErrorAction Stop
        if ((Get-FileHash -LiteralPath $download -Algorithm SHA256 -ErrorAction Stop).Hash -ne $expectedHash) {
            throw 'The downloaded arXivEdits training alignment has an unexpected SHA-256.'
        }
        Move-Item -LiteralPath $download -Destination $alignment -ErrorAction Stop
    }
    if ((Get-FileHash -LiteralPath $alignment -Algorithm SHA256 -ErrorAction Stop).Hash -ne $expectedHash) {
        throw 'The cached arXivEdits training alignment has an unexpected SHA-256.'
    }
    if (-not (Test-Path -LiteralPath "$data/experiment.json" -PathType Leaf -ErrorAction Stop)) {
        & $python scripts/expand_academic_pairs.py --version 3 --author-revisions $alignment
        if ($LASTEXITCODE -ne 0) { throw 'v3 dataset build failed. See the output above.' }
    }
    $manifest = Get-Content -LiteralPath "$data/manifest.json" -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    $manifest.counts | Format-List
}
```

The builder fits the v8 byte-level BPE tokenizer on the frozen v1 sentence training corpus and its word and grammar supplements. It checks that the frozen pairs fit the 192-token limit and edit encoding, then removes new pairs that fail those checks. The study requires at least **10,000 real training pairs** including the original 491. Its validation and test files are byte-for-byte copies of v1. The builder publishes `.cache/academic-pairs-v3` with the checked tokenizer and records its fingerprints in the manifest. Training verifies and uses those files. See [the v8 training and validation commands](LexiFAR_Training.md#train-on-the-author-revision-expansion).

## Loading and tokenization

Sentence training, validation and test records use `id`, `source` and `target`. The joint six-layer tokenizer is fitted to sentence and word training text only. Source and target receive BOS/EOS tokens, dynamic right padding uses attention masks, and target padding is ignored by cross-entropy. The versioned launcher verifies data hashes and rejects source/target truncation.

Use the included prepared files for normal training. Pinned raw downloading and data preparation are retained for provenance and creation of separately versioned studies; neither needs to run before using the existing corpus.
