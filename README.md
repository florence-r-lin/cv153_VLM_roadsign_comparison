# Investigating Effectiveness of Two Vision Language Models in Recognizing Embedded Texts in Road Signs

**CS153 Computer Vision Final Project**
Janav Shah and Florence Lin

View our final paper write up [here](https://github.com/florence-r-lin/cv153_VLM_roadsign_comparison)

---

## Overview

This project compares two Google Gemma vision language models being **Gemma 3 (4B-IT)** and **Gemma 4 (E4B-IT)** on their ability to read and interpret text embedded in road signs. We evaluate how good they are across five prompt types:

| Task | Prompt | Metric |
|---|---|---|
| Text extraction | "What text is written on this road sign?" | Exact match, BLEU |
| Meaning | "What does this road sign mean?" | Exact match, BLEU |
| Scenario: New Driver | "You are a new driver and see this sign at an intersection. What action do you take?" | BERTScore F1 |
| Scenario: Highway | "You are driving at highway speed and this sign appears ahead. What does it require you to do?" | BERTScore F1 |
| Scenario: Explain | "You are a passenger explaining this sign to someone who has never driven before. What does it mean and why does it matter?" | BERTScore F1 |

We evaluate on the [Kaggle Traffic Sign Dataset Classification](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification) which has 58 classes and 171 evaluation images.

One thing we found is that 67% of signs in this dataset are symbol-only (no readable text) so we split evaluation into **text signs** vs **symbol signs** for a better comparison.

---

## Repository Structure

```
cv153_final_project/
├── config.py                          # all paths, model IDs, prompts, sign categorization
├── requirements.txt                   # pip install requirements in venv
├── data/                              # download dataset via Kaggle (gitignored)
│   ├── processed/                     
│   └── traffic_signs/                 
├── notebooks/
│   └── demo.ipynb                     # full demo notebook for Prof to try it out
├── outputs/
│   ├── predictions/                   # model prediction CSVs (gitignored, generated)
│   └── metrics/                       # evaluation results (gitignored, generated)
└── scripts/
    ├── prepare_data.py                # builds eval_metadata.csv from raw dataset
    ├── run_gemma3.py                  # Gemma 3 inference on text_extraction and meaning
    ├── run_gemma4.py                  # Gemma 4 inference on text_extraction and meaning
    ├── run_scenario.py                # Part 2, scenario prompting
    ├── generate_references.py         # generates reference answers for BERTScore
    ├── evaluate.py                    # exact match and BLEU for text and meaning prompts
    └── evaluate_bertscore.py          # BERTScore F1 for scenario prompts
```

---

## Environment Setup

### Requirements
- HMC CS server access
- Python 3.11
- CUDA GPU 
- Kaggle account + API token
- HuggingFace account + API token (for Gemma 3 access, terms and conditions need to be accepted)

### 1: SSH into server and start tmux session
```bash
ssh YOUR_USERNAME@shadowfax.cs.hmc.edu
tmux new -s work
```

### 2: activate desired virtual environment and install dependecies
venv example:
```bash
source /cs/cs153/customenvs/YOUR_USERNAME/cs153finalproject/bin/activate
```

### 3: set up Jupyter kernel (for running project)
```bash
python -m ipykernel install --user --name cs153finalproject --display-name "cs153finalproject"
```

### 4: pick a GPU
```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0   # whichever has more free memory, sometimes 1 does but usually 0
```

### 5: download datasets

You will need a Kaggle API token:
1. Go to [kaggle.com](https://kaggle.com) → Settings → API → **Create New Token**
2. Set API token as an environment variable:
```bash
export KAGGLE_API_TOKEN=your_token_here
echo 'export KAGGLE_API_TOKEN=your_token_here' >> ~/.bashrc
```

Download Traffic Sign Classification Dataset:
```bash
mkdir -p /cs/cs153/projects/jshah/cv153_final_project/data/traffic_signs
cd /cs/cs153/projects/jshah/cv153_final_project/data/traffic_signs
kaggle datasets download -d ahemateja19bec1025/traffic-sign-dataset-classification
unzip -q traffic-sign-dataset-classification.zip
rm traffic-sign-dataset-classification.zip
```

### 6: log into HuggingFace

Gemma 3 requires accepting the license on HuggingFace before downloading:
Accept Gemma 3: https://huggingface.co/google/gemma-3-4b-it
Accept Gemma 4: https://huggingface.co/google/gemma-4-E4B-it

Log into HuggingFace:
```bash
hf auth login
```

### 7: install dependencies, make sure to add no-cache-dir
```bash
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir bert-score
```

---

## Reproducing Results

### 1: Build evaluation metadata
```bash
cd /cs/cs153/projects/jshah/cv153_final_project
python scripts/prepare_data.py
```

### 2: Run text extraction and meaning inference
```bash
python scripts/run_gemma3.py --prompt_type text_extraction
python scripts/run_gemma3.py --prompt_type meaning
python scripts/run_gemma4.py --prompt_type text_extraction
python scripts/run_gemma4.py --prompt_type meaning
```

### 4: Run scenario inference
```bash
python scripts/run_scenario.py --model gemma3
python scripts/run_scenario.py --model gemma4
```

### 5: Evaluate text extraction and meaning
```bash
python scripts/evaluate.py --prompt_type text_extraction
python scripts/evaluate.py --prompt_type meaning
```

### 6: Generate scenarior references and run BERTScore
```bash
python scripts/generate_references.py
python scripts/evaluate_bertscore.py
```

All the results are saved in `outputs/metrics/`.


## Results Summary

### Text Extraction (Exact Match %)
| Model | All Signs | Text Signs | Symbol Signs |
|---|---|---|---|
| Gemma 3 | 21.1% | 60.7% | 1.7% |
| Gemma 4 | 17.5% | 53.6% | 0.0% |

### Meaning Task (Avg BLEU)
| Model | All Signs | Text Signs | Symbol Signs |
|---|---|---|---|
| Gemma 3 | 0.0937 | 0.1124 | 0.0846 |
| Gemma 4 | 0.0916 | 0.1355 | 0.0702 |

### Scenario Prompts (Avg BERTScore F1)
| Model | New Driver | Highway | Explain |
|---|---|---|---|
| Gemma 3 | 0.8949 | 0.9037 | 0.8938 |
| Gemma 4 | 0.8917 | 0.8938 | 0.8943 |

---

## References

**Models**
To load both models, we are using borrowed code from HuggingFace. 
- Gemma Team, Google. (2024). Gemma 3: Open Models Based on Gemini Technology and Research. https://huggingface.co/google/gemma-3-4b-it
- Gemma Team, Google. (2024). Gemma 4. https://huggingface.co/google/gemma-4-E4B-it

**Literature Review and Background**
- Hwang, J.-J., et al. (2024). EMMA: End-to-end multimodal model for autonomous driving. *arXiv preprint arXiv:2410.23262*.
- Pandey, A., Bodo, D., Phukan, A., & Ekbal, A. (2025). The quest for visual understanding: A journey through the evolution of visual question answering. *arXiv preprint arXiv:2501.07109*.
- Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., et al. (2025). Gemma 3 technical report. Google DeepMind. *arXiv preprint arXiv:2504.07491*.
- Google DeepMind. (2026). Gemma 4. https://deepmind.google/models/gemma/gemma-4/

**Evaluation Metrics**
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the ACL*, 311–318.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). BERTScore: Evaluating text generation with BERT. *arXiv preprint arXiv:1904.09675*.
- Rajpurkar, P., et al. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *EMNLP*. *(exact match normalization pattern)*
- Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*. *(underlying model used by BERTScore)*
- Shi, T., & Wieting, J. (2019). BERTScore (Version 0.3.13) [Software]. PyPI. https://pypi.org/project/bert-score/
- Bird, S., Klein, E., & Loper, E. (n.d.). bleu_score [Software module]. Natural Language Toolkit (NLTK). https://www.nltk.org/_modules/nltk/translate/bleu_score.html

**Implementation**
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*, 33. *(zero-shot prompting)*
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*, 35. *(instruction-style prompting)*

**Datasets**
- Rafsan. (2024). US Road Signs Object Detection Model. Kaggle. https://www.kaggle.com/datasets/rafsanrudro/us-road-signs-object-detection-model-by-us

