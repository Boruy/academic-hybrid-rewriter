# Third-party data and libraries

## IteraTeR and synthetic training pairs

IteraTeR-HUMAN sentence data are from Wanyu Du, Vipul Raheja, Dhruv Kumar, Zae Myung Kim, Melissa Lopez and Dongyeop Kang, Understanding Iterative Revision from Human-Written Text (ACL 2022).

- Dataset: https://huggingface.co/datasets/wanyu/IteraTeR_human_sent
- Source repository: https://github.com/vipulraheja/iterater
- Pinned revision: `e22e0371dac444239b944f9293f5b491d62b73f0`

The upstream dataset card lists Apache-2.0. The upstream repository license is retained as `data/IteraTeR_LICENSE.txt`. Copies in `data/processed` are modified through normalization, arXiv/intent selection, length/number filtering, deduplication and document-group splitting. See `docs/Dataset_Card.md` and `data/raw/manifest.json`. Preserve attribution when using the text.

`data/academic-pairs-v1` combines the attributed real training split with 12,000 model-authored synthetic semantic-frame pairs. Synthetic sentences describe hypothetical findings rather than newly redistributed paper text or independent human revisions. Provenance, source hashes and limitations are retained in its README and manifest.

## Model and installed libraries

The six-layer model uses PyTorch modules and the project's plain-PyTorch Mamba reference recurrence. It trains from scratch and does not bundle pretrained model weights. PyTorch, Transformers, Tokenizers and NumPy are separately installed dependencies governed by their upstream licenses. Optimized official Mamba kernels are not bundled or validated here.

Optional overlap scoring uses SacreBLEU (Apache-2.0): https://github.com/mjpost/sacrebleu.

Checkpoints, prediction outputs, virtual environments and caches are excluded from the current Git tree. Historical experiment artifacts can still exist in Git history or the ignored local archive. No new license is assigned to the team's original code.

## Preserved arXivEdits references

Chao Jiang, Wei Xu and Samuel Stevens, arXivEdits: Understanding the Human Revision Process in Scientific Writing, EMNLP 2022: https://aclanthology.org/2022.emnlp-main.641/.

Source repository: https://github.com/chaojiang06/arXivEdits, pinned commit `113913d68522c1c8428d8e191d5c9bed633da6ce`.

Public curation records retain source references, hashes, intentions, partition decisions and paper-version license URLs. Original paper text remains in the ignored local cache. Mixed paper-specific licenses apply; public availability and the IteraTeR license do not grant a blanket license over this separate collection. Preserve the frozen partition, source fingerprints and paper-specific metadata. These references are not independently approved training examples. See `docs/Dataset_Card.md`.
