# Phase 3 - Colab T4 GPU Evaluation Instructions

## 🎯 Objective
Run comprehensive evaluation on Colab T4 GPU:
1. Full ablation study (n=97 cases)
2. Bootstrap confidence intervals for all metrics
3. BioMistral comparative evaluation

## 📋 Prerequisites
✅ Colab MCP connected  
✅ Code pushed to GitHub  
✅ T4 GPU runtime selected in Colab

## 🚀 Quick Start (Option A: Use Existing Notebook)

Open `evaluation/colab_runner.ipynb` in Colab and run:

### Cell 1: Configuration
```python
GITHUB_REPO = "https://github.com/kbssrikar7/healthcare-qa-chatbot.git"
PROJECT_DIR = "/content/project"
```

### Cell 2: Setup
```python
# Clone repo
!git clone {GITHUB_REPO} {PROJECT_DIR}
%cd {PROJECT_DIR}

# Install dependencies (minimal for evaluation)
!pip install -q torch transformers sentence-transformers
!pip install -q chromadb rank-bm25 rouge-score evaluate bert-score
!pip install -q loguru tqdm pandas numpy scikit-learn
```

### Cell 3: Run Phase 3 Evaluation
```python
!python evaluation/phase3_colab_eval.py
```

This will:
- Run full ablation study (n=97) → `evaluation/results/ablation.json`
- Compute metrics with bootstrap CIs → `evaluation/results/metrics_full_tinyllama.json`
- Test BioMistral → `evaluation/results/metrics_full_biomistral.json`

### Cell 4: Download Results
```python
from google.colab import files
import shutil

# Create archive
!cd evaluation/results && tar -czf /tmp/phase3_results.tar.gz *.json *.png

# Download
files.download('/tmp/phase3_results.tar.gz')
```

## 🚀 Alternative (Option B: Run Individual Scripts)

### 1. Full Ablation Study
```bash
python evaluation/run_ablation.py --n 97 --out-dir evaluation/results
```

**Output**: `evaluation/results/ablation.json`
- ECE per ablation variant
- Mean confidence per variant
- Signal contribution analysis

### 2. Bootstrap Confidence Intervals
```bash
python evaluation/run_paper_eval.py --mode metrics --n 97
```

**Output**: `evaluation/results/metrics_full_tinyllama.json`
- Keyword coverage with 95% CI
- ROUGE-L with 95% CI
- BERTScore with 95% CI
- Statistical significance metrics

### 3. BioMistral Investigation
```bash
python evaluation/run_paper_eval.py --mode metrics --model biomistral --n 20
```

**Output**: `evaluation/results/metrics_full_biomistral.json`
- BioMistral performance comparison
- Identifies prompt format issues

## 📊 Expected Runtime (T4 GPU)
- Ablation study (n=97): ~30-40 minutes
- Full metrics evaluation: ~45-60 minutes  
- BioMistral test (n=20): ~20-30 minutes
- **Total: ~1.5-2 hours**

## 📥 Results to Retrieve
After completion, download these files:
```
evaluation/results/
├── ablation.json              # Full ablation study
├── metrics_full_tinyllama.json   # Metrics with bootstrap CIs
├── metrics_full_biomistral.json  # BioMistral comparison
├── calibration.json           # Updated calibration params
└── *.png                      # Generated figures
```

## 🔄 Sync Results Back to Local

After downloading results from Colab:

```bash
# Extract archive
cd /home/kbs/Documents/final_project
tar -xzf ~/Downloads/phase3_results.tar.gz -C evaluation/results/

# Commit results
git add evaluation/results/
git commit -m "feat: Phase 3 evaluation results from Colab T4

- Full ablation study (n=97)
- Bootstrap confidence intervals
- BioMistral comparative analysis"
```

## 🐛 Troubleshooting

### Issue: "Module not found"
```python
import sys
sys.path.insert(0, '/content/project')
```

### Issue: "CUDA out of memory"
```python
# Reduce batch size
os.environ["BATCH_SIZE"] = "4"
```

### Issue: "ChromaDB not found"
The evaluation scripts work without the full KB - they use cached responses from `data/cache/` if available.

## ✅ Verification

After completion, verify results:
```python
import json

# Check ablation results
with open('evaluation/results/ablation.json') as f:
    ablation = json.load(f)
    print(f"Ablation variants tested: {len(ablation)}")
    print(f"Sample size: {ablation.get('n', 'N/A')}")

# Check metrics
with open('evaluation/results/metrics_full_tinyllama.json') as f:
    metrics = json.load(f)
    print(f"Keyword coverage: {metrics.get('keyword_coverage_mean', 'N/A')}")
    print(f"ROUGE-L: {metrics.get('rougeL_mean', 'N/A')}")
    print(f"BERTScore: {metrics.get('bertscore_f1_mean', 'N/A')}")
```

## 📝 Notes

- All scripts support `--n` parameter to limit test cases during debugging
- Results are automatically saved with timestamps
- Baseline metrics preserved in `evaluation/results/_snapshots/2026-04-07/`
- Use `HF_HUB_OFFLINE=1` to avoid network delays

---

**Status**: Ready for execution  
**Updated**: 2026-04-07  
**Estimated completion**: 1.5-2 hours on T4 GPU
