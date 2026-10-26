# Third party data and libraries

IteraTeR-HUMAN sentence data are from Wanyu Du, Vipul Raheja, Dhruv Kumar, Zae Myung Kim, Melissa Lopez and Dongyeop Kang, Understanding Iterative Revision from Human-Written Text (ACL 2022).

Dataset: https://huggingface.co/datasets/wanyu/IteraTeR_human_sent
Repository: https://github.com/vipulraheja/iterater
Pinned revision: e22e0371dac444239b944f9293f5b491d62b73f0
The upstream dataset card lists Apache-2.0. The upstream repository license is retained as data/IteraTeR_LICENSE.txt.
Processed copies in data/processed are modified versions: normalization, arXiv/intent selection, length/number filtering, deduplication and document-group re-splitting. See docs/Dataset_Card.md and data/raw/manifest.json for provenance and counts. Preserve the original-source attributions when using the text.

The repository contains locally generated tiny BART-format smoke checkpoints and fine-tuned checkpoints initialized from facebook/bart-base. The BART model card is available at https://huggingface.co/facebook/bart-base. Preserve the upstream model attribution and applicable terms. PyTorch, Transformers and Tokenizers are separately installed dependencies under their respective upstream licenses. The Mamba reference adapter in src is newly written for this milestone. The optional official mamba_ssm package is not bundled or validated here.

Raw and processed data, experiment records and fine-tuned checkpoint binaries are included in this repository. Model weights use Git LFS. Virtual environments and the local Hugging Face cache are excluded because they are installation dependencies rather than portable project files. The repository does not assign a new license to the team’s original code.
