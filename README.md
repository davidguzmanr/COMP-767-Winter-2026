# COMP-767-Winter-2026
Project for COMP-767: Large Language Models

## Culture Drifts: How Post-Training Shapes Cultural Awareness in Vision-Language Models

### Datasets

#### SFT / Evaluation
| Dataset | HF ID | Size | Coverage |
|---|---|---|---|
| CulturalGround | [neulab/CulturalGround](https://huggingface.co/datasets/neulab/CulturalGround) | 22M VQA | 42 countries, 39 languages |
| CVQA | [afaji/cvqa](https://huggingface.co/datasets/afaji/cvqa) | 10K | 30 countries, 31 languages |
| CulturalVQA | [mair-lab/CulturalVQA](https://huggingface.co/datasets/mair-lab/CulturalVQA) | 2.4K | 11 countries, 5 continents |
| Culture Affordance Atlas | None | 367 obj-func pairs | 63 countries (eval only) |
| Seeing Culture | [Multimedia-SMU/seeingculture-benchmark](https://huggingface.co/datasets/Multimedia-SMU/seeingculture-benchmark) | 3K Q | 7 SE Asian countries (eval only) |

#### DPO (preference pairs)
Synthetic preference dataset constructed from CVQA/CulturalVQA: chosen = culturally grounded answer, rejected = response from a base model prompted without cultural conditioning. See `scripts/data/preprocess_datasets.py`.

General-purpose VLM preference datasets (pipeline validation):
| Dataset | HF ID | Size | Notes |
|---|---|---|---|
| RLAIF-V (formatted) | [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) | 83K | Pre-formatted for TRL DPOTrainer |
| RLAIF-V (raw) | [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) | ~100K | Sources: COCO, TextVQA, GQA, ShareGPT4V |
| RLHF-V | [openbmb/RLHF-V-Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset) | 5.7K | Fine-grained human feedback |
| MM-RLHF | [yifanzhang114/MM-RLHF](https://huggingface.co/datasets/yifanzhang114/MM-RLHF) | 16K | Expert-ranked; diverse visual domains |
