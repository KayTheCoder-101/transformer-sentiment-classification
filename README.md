# Transformer NLP: Sentiment Classification with Attention Analysis, SHAP & LIME

> **Assignment:** Transformers, Attention Layer, SHAP and LIME — Amazon Polarity Dataset

---

## Project Overview

This project fine-tunes **BERT-base-uncased** on the [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) dataset for binary sentiment classification (Positive / Negative). It includes:

- Full data pipeline (cleaning, tokenization, stratified splitting)
- BERT fine-tuning with attention output enabled
- Attention heatmap visualization across layers and heads
- SHAP and LIME explanations for 20+ predictions each
- Comparative analysis of SHAP vs LIME (faithfulness, stability, runtime)
- Error analysis of misclassified samples

---

## Repository Structure

```
.
├── transformer_nlp_assignment.ipynb   ← Main notebook (all code)
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
└── outputs/                           ← Generated figures (after running notebook)
    ├── eda_overview.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── attention_heatmaps.png
    ├── attention_layer_rollout.png
    ├── shap_20_samples.png
    ├── lime_20_samples.png
    ├── shap_lime_comparison.png
    ├── error_analysis_lime.png
    ├── methodology_flowchart.png
    └── best_model.pt
```

---

## Hardware & Software Environment

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | NVIDIA RTX 3090 / A100 (24GB VRAM) | Any CUDA GPU (8GB VRAM) |
| **RAM** | 16 GB | 8 GB |
| **Disk** | 10 GB free | 5 GB free |
| **Python** | 3.10+ | 3.8+ |
| **CUDA** | 11.8 / 12.1 | 11.0 |
| **OS** | Ubuntu 22.04 / Windows 11 | Any |

> **CPU fallback:** The notebook runs on CPU but training will take ~5–10× longer. Reduce `N_TRAIN` to 5,000 and `EPOCHS` to 2 for a quick test run.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation

The dataset is downloaded automatically from HuggingFace on first run. No manual download required.

```python
# Handled inside the notebook:
from datasets import load_dataset
raw = load_dataset("fancyzhx/amazon_polarity")
```

If you are behind a proxy or firewall:

```bash
HF_DATASETS_OFFLINE=0 python -c "from datasets import load_dataset; load_dataset('fancyzhx/amazon_polarity')"
```

---

## Running the Notebook

### Option A — Jupyter Lab (recommended)

```bash
jupyter lab transformer_nlp_assignment.ipynb
```

Then run all cells: **Kernel → Restart & Run All**

### Option B — Jupyter Notebook classic

```bash
jupyter notebook transformer_nlp_assignment.ipynb
```

### Option C — Command line (convert to script)

```bash
jupyter nbconvert --to script transformer_nlp_assignment.ipynb
python transformer_nlp_assignment.py
```

### Option D — Google Colab

1. Upload `transformer_nlp_assignment.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells

---

## Configuration

Edit the configuration block at the top of **Section 1.2** to adjust resource usage:

```python
N_TRAIN  = 20_000   # training samples  (reduce to 5_000 for quick test)
N_VAL    = 4_000    # validation samples
N_TEST   = 4_000    # test samples
MAX_LEN  = 128      # BERT token length (max 512)
EPOCHS   = 3        # training epochs   (reduce to 1 for quick test)
LR       = 2e-5     # learning rate
```

---

## Expected Results

| Metric | Expected Range |
|--------|----------------|
| Test Accuracy | 92–95% |
| Test Precision | 91–95% |
| Test Recall | 92–96% |
| Test F1-Score | 92–95% |

*Exact values depend on GPU, random seed, and subset size.*

---

## Section Breakdown

| Section | Description |
|---------|-------------|
| **0** | Environment setup & hardware info |
| **1** | Data pipeline: loading, cleaning, sampling, tokenization |
| **2** | BERT fine-tuning, training loop, test evaluation |
| **3** | Attention extraction and heatmap visualization |
| **4** | SHAP explanations (20 samples) |
| **5** | LIME explanations (20 samples) |
| **6** | Comparative analysis: faithfulness, stability, runtime |
| **7** | Error analysis: misclassified samples |
| **8** | End-to-end methodology flowchart |
| **9** | Final summary table |

---

## Reproducing Results

To reproduce exact results, ensure:

1. The random seed is unchanged: `SEED = 42`
2. The same model checkpoint is used: `bert-base-uncased`
3. Dependencies match `requirements.txt`

The notebook saves `best_model.pt` at the end of training. To skip training and load a saved model:

```python
model.load_state_dict(torch.load('best_model.pt', map_location=DEVICE))
```

---

## References

1. Dataset: https://huggingface.co/datasets/fancyzhx/amazon_polarity
2. SHAP: https://shap.readthedocs.io/en/latest/text_examples.html
3. LIME: https://lime-ml.readthedocs.io/en/latest/
4. BERT: Devlin et al. (2019) — https://arxiv.org/abs/1810.04805
5. HuggingFace Transformers: https://huggingface.co/docs/transformers
