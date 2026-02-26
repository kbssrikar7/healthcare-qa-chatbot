#!/usr/bin/env python3
"""Generate BioMistral Evaluation Colab notebook."""
import json
from pathlib import Path

def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(source_str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source_str]}

cells = []

# Title
cells.append(md([
    "# Healthcare QA Chatbot - BioMistral-7B Evaluation\n",
    "\n",
    "Evaluate **BioMistral-7B** (medical-specialized) vs **TinyLlama-1.1B** using Colab T4 GPU.\n",
    "\n",
    "> **Set runtime to GPU first:** Runtime > Change runtime type > T4 GPU\n",
]))

# 1. GPU check
cells.append(md(["## 1. GPU Check & Install Dependencies"]))
cells.append(code(
    "import torch\n"
    "assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU'\n"
    "print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)')\n"
))
cells.append(code(
    "!pip install -q transformers>=4.36.0 bitsandbytes>=0.41.0 accelerate>=0.25.0 chromadb==0.4.24 sentence-transformers>=2.2.0 rank-bm25>=0.2.2 pandas tqdm\n"
    "print('Dependencies installed!')\n"
))

# 2. Upload KB
cells.append(md([
    "## 2. Upload Knowledge Base\n",
    "\n",
    "On your local machine first:\n",
    "```bash\n",
    "cd ~/Documents/final_project\n",
    "zip -r knowledge_base.zip data/knowledge_base/\n",
    "```\n",
    "Then run the cell below and upload.\n",
]))
cells.append(code(
    "import os, zipfile, shutil\n"
    "from pathlib import Path\n"
    "from google.colab import files\n"
    "\n"
    "KB_DIR = Path('/content/data/knowledge_base')\n"
    "if KB_DIR.exists() and any(KB_DIR.iterdir()):\n"
    "    print(f'Knowledge base already at {KB_DIR}')\n"
    "else:\n"
    "    print('Upload your knowledge_base.zip...')\n"
    "    uploaded = files.upload()\n"
    "    for fname in uploaded:\n"
    "        with zipfile.ZipFile(fname, 'r') as z:\n"
    "            z.extractall('/content/')\n"
    "    if not KB_DIR.exists():\n"
    "        for p in Path('/content').rglob('chroma.sqlite3'):\n"
    "            KB_DIR.mkdir(parents=True, exist_ok=True)\n"
    "            for item in p.parent.iterdir():\n"
    "                dest = KB_DIR / item.name\n"
    "                if item.is_dir(): shutil.copytree(item, dest, dirs_exist_ok=True)\n"
    "                else: shutil.copy2(item, dest)\n"
    "            break\n"
    "    print(f'Extracted to {KB_DIR}: {[f.name for f in KB_DIR.iterdir()]}')\n"
))

# 3. Retrieval
cells.append(md(["## 3. Initialize Retrieval Pipeline"]))
cells.append(code(
    "import chromadb\n"
    "from sentence_transformers import SentenceTransformer\n"
    "\n"
    "embedder = SentenceTransformer('all-MiniLM-L6-v2')\n"
    "client = chromadb.PersistentClient(path=str(KB_DIR))\n"
    "collection = client.get_collection('medical_knowledge')\n"
    "print(f'Knowledge base: {collection.count():,} documents')\n"
    "\n"
    "def retrieve(query, top_k=5):\n"
    "    qe = embedder.encode([query])[0].tolist()\n"
    "    r = collection.query(query_embeddings=[qe], n_results=top_k)\n"
    "    return [{'content': d, 'source': m.get('source','?'), 'score': round(1-dist,4)}\n"
    "            for d, m, dist in zip(r['documents'][0], r['metadatas'][0], r['distances'][0])]\n"
    "\n"
    "test = retrieve('What are the symptoms of diabetes?')\n"
    "print(f'Test: {len(test)} passages, top score: {test[0][\"score\"]:.3f}')\n"
))

# 4. Load models
cells.append(md(["## 4. Load Models"]))
cells.append(code(
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
    "import time\n"
    "\n"
    "# BioMistral-7B (4-bit quantized)\n"
    "print('Loading BioMistral-7B with 4-bit quantization...')\n"
    "t0 = time.time()\n"
    "bnb_config = BitsAndBytesConfig(\n"
    "    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,\n"
    "    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')\n"
    "bio_tok = AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B', trust_remote_code=True)\n"
    "bio_model = AutoModelForCausalLM.from_pretrained(\n"
    "    'BioMistral/BioMistral-7B', quantization_config=bnb_config,\n"
    "    device_map='auto', trust_remote_code=True)\n"
    "bio_model.eval()\n"
    "if bio_tok.pad_token is None: bio_tok.pad_token = bio_tok.eos_token\n"
    "print(f'BioMistral loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB')\n"
))
cells.append(code(
    "# TinyLlama-1.1B\n"
    "print('Loading TinyLlama-1.1B...')\n"
    "t0 = time.time()\n"
    "tiny_tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', trust_remote_code=True)\n"
    "tiny_model = AutoModelForCausalLM.from_pretrained(\n"
    "    'TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float16,\n"
    "    device_map='auto', trust_remote_code=True)\n"
    "tiny_model.eval()\n"
    "if tiny_tok.pad_token is None: tiny_tok.pad_token = tiny_tok.eos_token\n"
    "print(f'TinyLlama loaded in {time.time()-t0:.1f}s, Total GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB')\n"
))

# 5. Generation functions
cells.append(md(["## 5. Generation & Cleaning Functions"]))
cells.append(code(
    "import re\n"
    "\n"
    "STOP_PATTERNS = [\n"
    "    r'\\nQuestion:', r'\\nQ:', r'\\nAnswer:', r'Best regards', r'Sincerely',\n"
    "    r'ChatDoctor', r'HealthCareMagic', r'Thank you for', r'Take care',\n"
    "    r'I hope this', r'\\nDear ', r'\\n---',\n"
    "]\n"
    "\n"
    "def clean(text):\n"
    "    if not text: return text\n"
    "    text = text.strip()\n"
    "    for p in ['Answer:', 'Factual Answer:', 'Based on the reference text,']:\n"
    "        if text.startswith(p): text = text[len(p):].strip()\n"
    "    cut = min(50, len(text))\n"
    "    best = len(text)\n"
    "    for pat in STOP_PATTERNS:\n"
    "        m = re.search(pat, text, re.IGNORECASE)\n"
    "        if m and m.start() >= cut and m.start() < best: best = m.start()\n"
    "    return re.sub(r'\\[\\d+\\]', '', text[:best]).strip()\n"
    "\n"
    "def generate(question, context, model, tokenizer, max_tokens=256):\n"
    "    \"\"\"Generate RAG answer. Returns (text, latency_s, n_tokens).\"\"\"\n"
    "    SYS = chr(60) + '|system|' + chr(62)\n"
    "    USR = chr(60) + '|user|' + chr(62)\n"
    "    AST = chr(60) + '|assistant|' + chr(62)\n"
    "    END = chr(60) + '/s' + chr(62)\n"
    "    prompt = (\n"
    "        f'{SYS}\\n'\n"
    "        'Answer the question using ONLY the reference text. '\n"
    "        'Do NOT add your own knowledge. Be concise.\\n'\n"
    "        f'{END}\\n'\n"
    "        f'{USR}\\n'\n"
    "        f'REFERENCE TEXT: {context}\\n\\n'\n"
    "        f'QUESTION: {question}\\n'\n"
    "        f'{END}\\n'\n"
    "        f'{AST}\\n'\n"
    "    )\n"
    "    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)\n"
    "    input_len = inputs.input_ids.shape[1]\n"
    "    t0 = time.time()\n"
    "    with torch.no_grad():\n"
    "        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3,\n"
    "                             top_p=0.85, do_sample=True, pad_token_id=tokenizer.pad_token_id)\n"
    "    latency = time.time() - t0\n"
    "    gen_ids = out[0][input_len:]\n"
    "    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)\n"
    "    return clean(answer), latency, len(gen_ids)\n"
))

# 6. Evaluation
cells.append(md(["## 6. Run Evaluation - BioMistral vs TinyLlama"]))
cells.append(code(
    "import pandas as pd\n"
    "from tqdm.notebook import tqdm\n"
    "\n"
    "EVAL_QUESTIONS = [\n"
    "    'What are the symptoms of type 2 diabetes?',\n"
    "    'What causes hypertension?',\n"
    "    'What is the treatment for asthma?',\n"
    "    'What are the side effects of metformin?',\n"
    "    'How is pneumonia diagnosed?',\n"
    "    'What is the difference between Type 1 and Type 2 diabetes?',\n"
    "    'What are the risk factors for heart disease?',\n"
    "    'How does aspirin work as a blood thinner?',\n"
    "    'What is COPD and how is it treated?',\n"
    "    'What are the symptoms of a stroke?',\n"
    "]\n"
    "\n"
    "results = []\n"
    "\n"
    "for q in tqdm(EVAL_QUESTIONS, desc='Evaluating'):\n"
    "    # Retrieve context\n"
    "    passages = retrieve(q, top_k=5)\n"
    "    context = '\\n\\n'.join([p['content'] for p in passages[:3]])\n"
    "    avg_retrieval_score = sum(p['score'] for p in passages) / len(passages)\n"
    "\n"
    "    # Generate with BioMistral\n"
    "    bio_answer, bio_latency, bio_tokens = generate(q, context, bio_model, bio_tok)\n"
    "\n"
    "    # Generate with TinyLlama\n"
    "    tiny_answer, tiny_latency, tiny_tokens = generate(q, context, tiny_model, tiny_tok)\n"
    "\n"
    "    results.append({\n"
    "        'question': q,\n"
    "        'retrieval_score': round(avg_retrieval_score, 4),\n"
    "        'biomistral_answer': bio_answer,\n"
    "        'biomistral_latency_s': round(bio_latency, 2),\n"
    "        'biomistral_tokens': bio_tokens,\n"
    "        'tinyllama_answer': tiny_answer,\n"
    "        'tinyllama_latency_s': round(tiny_latency, 2),\n"
    "        'tinyllama_tokens': tiny_tokens,\n"
    "    })\n"
    "    print(f'\\nQ: {q}')\n"
    "    print(f'  BioMistral ({bio_latency:.1f}s): {bio_answer[:120]}...')\n"
    "    print(f'  TinyLlama  ({tiny_latency:.1f}s): {tiny_answer[:120]}...')\n"
    "\n"
    "df = pd.DataFrame(results)\n"
    "print(f'\\nEvaluation complete! {len(df)} questions evaluated.')\n"
))

# 7. Summary table
cells.append(md(["## 7. Results Summary"]))
cells.append(code(
    "# Summary statistics\n"
    "print('=' * 60)\n"
    "print('MODEL COMPARISON SUMMARY')\n"
    "print('=' * 60)\n"
    "print(f'\\nBioMistral-7B:')\n"
    "print(f'  Avg latency: {df[\"biomistral_latency_s\"].mean():.2f}s')\n"
    "print(f'  Avg tokens:  {df[\"biomistral_tokens\"].mean():.0f}')\n"
    "print(f'  Avg answer length: {df[\"biomistral_answer\"].str.len().mean():.0f} chars')\n"
    "print(f'\\nTinyLlama-1.1B:')\n"
    "print(f'  Avg latency: {df[\"tinyllama_latency_s\"].mean():.2f}s')\n"
    "print(f'  Avg tokens:  {df[\"tinyllama_tokens\"].mean():.0f}')\n"
    "print(f'  Avg answer length: {df[\"tinyllama_answer\"].str.len().mean():.0f} chars')\n"
    "print(f'\\nRetrieval:')\n"
    "print(f'  Avg score: {df[\"retrieval_score\"].mean():.4f}')\n"
    "print('=' * 60)\n"
    "\n"
    "# Show full table\n"
    "df[['question', 'biomistral_latency_s', 'tinyllama_latency_s', 'retrieval_score']]\n"
))

# 8. Export
cells.append(md(["## 8. Export Results"]))
cells.append(code(
    "# Save as CSV\n"
    "df.to_csv('/content/biomistral_vs_tinyllama_eval.csv', index=False)\n"
    "print('Saved to /content/biomistral_vs_tinyllama_eval.csv')\n"
    "\n"
    "# Download\n"
    "from google.colab import files\n"
    "files.download('/content/biomistral_vs_tinyllama_eval.csv')\n"
    "print('Download started!')\n"
))

# 9. Detailed comparison
cells.append(md(["## 9. Side-by-Side Answer Comparison"]))
cells.append(code(
    "# Display all answers side by side\n"
    "for _, row in df.iterrows():\n"
    "    print('=' * 80)\n"
    "    print(f'Q: {row[\"question\"]}')\n"
    "    print(f'Retrieval Score: {row[\"retrieval_score\"]}')\n"
    "    print(f'\\n--- BioMistral-7B ({row[\"biomistral_latency_s\"]}s) ---')\n"
    "    print(row['biomistral_answer'])\n"
    "    print(f'\\n--- TinyLlama-1.1B ({row[\"tinyllama_latency_s\"]}s) ---')\n"
    "    print(row['tinyllama_answer'])\n"
    "    print()\n"
))

# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = Path(__file__).parent.parent / "notebooks" / "BioMistral_Evaluation_Colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=4)

print(f"Generated: {out_path}")
print(f"Cells: {len(cells)}")
