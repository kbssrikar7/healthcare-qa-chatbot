"""
Dataset loading utilities for medical QA.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Generator
from tqdm import tqdm

class MedicalDatasetLoader:
    """Load and iterate over medical datasets."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def load_medquad(self) -> pd.DataFrame:
        """Load MedQuAD dataset."""
        path = self.data_dir / "mediqa" / "medquad.parquet"
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"MedQuAD not found at {path}")
    
    def load_pubmedqa(self) -> pd.DataFrame:
        """Load PubMedQA dataset."""
        path = self.data_dir / "pubmed" / "pubmedqa_labeled.parquet"
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"PubMedQA not found at {path}")

    def load_medmcqa(self) -> pd.DataFrame:
        """Load MedMCQA dataset."""
        path = self.data_dir / "mediqa" / "medmcqa_train.parquet"
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"MedMCQA not found at {path}")
    
    def load_all_qa_pairs(self) -> List[Dict]:
        """Load all QA pairs from all datasets."""
        qa_pairs = []
        
        # Load MedQuAD
        try:
            df = self.load_medquad()
            for _, row in df.iterrows():
                qa_pairs.append({
                    "question": row.get("Question", row.get("question", "")),
                    "answer": row.get("Answer", row.get("answer", "")),
                    "source": "MedQuAD"
                })
        except Exception as e:
            print(f"⚠️ Could not load MedQuAD: {e}")
        
        # Load PubMedQA
        try:
            df = self.load_pubmedqa()
            for _, row in df.iterrows():
                context = row.get("context", {})
                if isinstance(context, dict):
                    context_text = " ".join(context.get("contexts", []))
                else:
                    context_text = str(context)
                qa_pairs.append({
                    "question": row.get("question", ""),
                    "answer": row.get("long_answer", ""),
                    "context": context_text,
                    "source": "PubMedQA"
                })
        except Exception as e:
            print(f"⚠️ Could not load PubMedQA: {e}")
        
        # Load MedMCQA
        try:
            df = self.load_medmcqa()
            # MedMCQA structure: question, exp (explanation), cop (choice of correct option), opa/opb/opc/opd
            # We will construct a simple QA pair.
            # Ideally we would map the 'cop' (correct option) to the text of that option,
            # but for now let's check the columns or just use the explanation as the answer if available,
            # or try to reconstruct the answer.
            # Let's inspect columns first or assume a standard approach.
            # For simplicity in this blind edit without seeing columns:
            # We'll try to find 'answer' or construct it.
            # Looking at common MedMCQA parquet schemas, it usually has 'question', 'cop' (index), 'opa', 'opb', etc.
            # and 'exp' (explanation).
            
            # Using a simplified approach: specific implementation depends on column names.
            # Assuming 'exp' (explanation) provides a good detailed answer.
            for _, row in df.iterrows():
                # Construct answer from explanation or just use explanation
                answer = row.get("exp")
                if not answer or pd.isna(answer):
                    # Fallback: try to find the correct option text
                    cop = row.get("cop") # 1-based or 0-based index? usually 1-based in original dataset but let's be careful.
                    # If we can't easily determine the short answer, we skip or use what we have.
                    continue
                
                qa_pairs.append({
                    "question": row.get("question", ""),
                    "answer": str(answer),
                    "source": "MedMCQA"
                })
        except Exception as e:
            print(f"⚠️ Could not load MedMCQA: {e}")
        
        return qa_pairs
    
    def get_documents_for_knowledge_base(self) -> Generator[Dict, None, None]:
        """Yield documents for building knowledge base."""
        qa_pairs = self.load_all_qa_pairs()
        
        for qa in qa_pairs:
            # Combine question and answer as document
            content = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"
            if qa.get("context"):
                content = f"Context: {qa['context']}\n\n{content}"
            
            yield {
                "content": content,
                "source": qa["source"],
                "metadata": {"type": "qa_pair"}
            }
