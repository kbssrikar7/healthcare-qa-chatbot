# 🚀 Phase 3 Colab Execution - Quick Start Guide

## ✅ Everything is Ready!

All code has been committed locally. Here's how to execute Phase 3 on your Colab T4 GPU:

---

## 📋 Option 1: Use the Automated Notebook (Recommended)

### Step 1: Open Colab
1. Go to: https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload: `/home/kbs/Documents/final_project/evaluation/phase3_colab_automated.ipynb`

### Step 2: Set Runtime to T4 GPU
1. Click "Runtime" → "Change runtime type"
2. Select "T4 GPU"
3. Click "Save"

### Step 3: Run All Cells
1. Click "Runtime" → "Run all"
2. When prompted for Drive access, click "Connect"
3. Wait ~1.5-2 hours for completion

**That's it!** The notebook will:
- Clone your repo (latest code)
- Install dependencies
- Run all Phase 3 evaluations
- Save results to `evaluation/results_copilot/`
- Push results back to GitHub
- Download results archive

---

## 📋 Option 2: Manual Execution

If you prefer to run commands manually in Colab:

### Setup (Run Once)
```python
# Clone repo
!git clone https://github.com/kbssrikar7/healthcare-qa-chatbot.git /content/project
%cd /content/project

# Configure git
!git config user.name "kbssrikar7"
!git config user.email "kbssrikar7@gmail.com"

# Install dependencies
!pip install -q torch transformers sentence-transformers rouge-score evaluate bert-score
!pip install -q pandas numpy scikit-learn tqdm loguru chromadb rank-bm25
```

### Run Phase 3 Tasks
```python
import sys
sys.path.insert(0, '/content/project')

# Task 1: Full Ablation Study (n=97)
!python evaluation/run_ablation.py --n 97 --out-dir evaluation/results

# Task 2: Metrics with Bootstrap CIs
!python evaluation/run_paper_eval.py --mode metrics --n 97

# Task 3: BioMistral Test
!python evaluation/run_paper_eval.py --mode metrics --model biomistral --n 20
```

### Save Results
```python
# Create results directory
!mkdir -p evaluation/results_copilot/$(date +%Y%m%d_%H%M%S)

# Copy results
!cp evaluation/results/*.json evaluation/results_copilot/$(date +%Y%m%d_%H%M%S)/

# Download
from google.colab import files
!tar -czf /tmp/results.tar.gz evaluation/results_copilot/
files.download('/tmp/results.tar.gz')
```

---

## 📥 After Colab Completes

### On Your Local Machine:

```bash
cd /home/kbs/Documents/final_project

# If you used the automated notebook, results are already pushed
# Just pull them:
git pull origin main

# If you downloaded manually, extract:
tar -xzf ~/Downloads/results.tar.gz -C evaluation/

# Commit if not auto-pushed:
git add evaluation/results_copilot/
git commit -m "feat: Phase 3 evaluation results from Colab T4"
git push origin main
```

---

## 🎯 Expected Results

After completion, you should have:

```
evaluation/results_copilot/
└── 20260407_HHMMSS_phase3_colab/
    ├── ablation.json                    # Full ablation study (n=97)
    ├── metrics_full_tinyllama.json     # TinyLlama with 95% CIs
    ├── metrics_full_biomistral.json    # BioMistral comparison
    ├── calibration.json                 # Updated Platt parameters
    ├── execution_summary.json           # Run metadata
    └── *.png                            # Generated figures
```

---

## 🔍 Verify Results

```python
import json

# Check ablation
with open('evaluation/results_copilot/.../ablation.json') as f:
    data = json.load(f)
    print(f"Ablation tested: n={data.get('n')} cases")

# Check metrics  
with open('evaluation/results_copilot/.../metrics_full_tinyllama.json') as f:
    metrics = json.load(f)
    print(f"Keyword: {metrics['keyword_coverage_mean']:.3f}")
    print(f"ROUGE-L: {metrics['rougeL_mean']:.3f}")
    print(f"BERTScore: {metrics['bertscore_f1_mean']:.3f}")
```

---

## ⏱️ Estimated Times (T4 GPU)

- Ablation study (n=97): 30-40 minutes
- Full metrics evaluation: 45-60 minutes
- BioMistral test (n=20): 20-30 minutes
- **Total: ~1.5-2 hours**

---

## 🆘 Troubleshooting

### Issue: "No module named 'src'"
```python
import sys
sys.path.insert(0, '/content/project')
```

### Issue: "CUDA out of memory"
```python
# Restart runtime and reduce batch size
import os
os.environ["BATCH_SIZE"] = "4"
```

### Issue: Git push fails
The automated notebook handles authentication. If manual push needed:
```bash
git remote set-url origin https://kbssrikar7:YOUR_TOKEN@github.com/kbssrikar7/healthcare-qa-chatbot.git
git push
```

---

## ✅ Next Steps After Phase 3

Once results are complete, we'll proceed to:
- **Phase 4**: Documentation & architecture diagrams
- **Phase 5**: Human evaluation protocol  
- **Phase 6**: Literature review & paper draft

---

**Status**: ✅ Ready for execution  
**Notebook**: `evaluation/phase3_colab_automated.ipynb`  
**Estimated completion**: 1.5-2 hours on T4 GPU

🚀 **You can start now!** Upload the notebook to Colab and click "Run all"
