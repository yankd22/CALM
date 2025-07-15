## CALM

A repository of ['CALM: Consensus-Aware Localized Merging for Multi-Task Learning'](https://openreview.net/forum?id=OgfvSDn73E&noteId=OgfvSDn73E). ICML, 2025.

### Abstract

Model merging aims to integrate the strengths of multiple fine-tuned models into a unified model while preserving task-specific capabilities. Existing methods, represented by task arithmetic, are typically classified into global- and local-aware methods. However, global-aware methods inevitably cause parameter interference, while local-aware methods struggle to maintain the effectiveness of task-specific details in the merged model. To address these limitations, we propose a Consensus Aware Localized Merging (CALM) method which incorporates localized information aligned with global task consensus, ensuring its effectiveness post-merging. CALM consists of three key components: (1) class-balanced entropy minimization sampling, providing a more flexible and reliable way to leverage unsupervised data; (2) an efficient-aware framework, selecting a small set of tasks for sequential merging with high scalability; (3) a consensus-aware mask optimization, aligning localized binary masks with global task consensus and merging them conflict-free. Experiments demonstrate the superiority and robustness of our CALM, significantly outperforming existing methods and achieving performance close to traditional MTL.

### Checkpoints
You can download the fine-tuned checkpoints from [task vectors](https://github.com/mlfoundations/task_vectors) or [the Google Drive folder](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw?usp=share_link).

### Data
You can also download the data processed by [task vectors](https://github.com/mlfoundations/task_vectors) from HuggingFace.

### Code
bash run.sh

### Acknowledgement
The codebase is built upon [AdaMerging](https://github.com/EnnengYang/AdaMerging) and [Localize-and-Stitch](https://github.com/uiuctml/Localize-and-Stitch), which also references code from [Task Arithmetic](https://github.com/mlfoundations/task_vectors), [TIES-MERGING](https://github.com/prateeky2806/ties-merging) and [Model Soups](https://github.com/mlfoundations/model-soups).
