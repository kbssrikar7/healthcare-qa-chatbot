"""
Dataset loading utilities for medical QA.
Supports: MedQuAD, PubMedQA, MedMCQA, HealthCareMagic, MedQA USMLE,
ChatDoctor, Medical Meadow, and additional consolidated datasets.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional

import pandas as pd
from loguru import logger


class MedicalDatasetLoader:
    """Load and iterate over medical datasets."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def _safe_load_parquet(self, path: Path) -> Optional[pd.DataFrame]:
        """Safely load a parquet file."""
        if path.exists():
            return pd.read_parquet(path)
        return None

    # ==================== Core Datasets ====================

    def load_medquad(self) -> pd.DataFrame:
        """Load MedQuAD dataset."""
        path = self.data_dir / "mediqa" / "medquad.parquet"
        df = self._safe_load_parquet(path)
        if df is not None:
            return df
        raise FileNotFoundError(f"MedQuAD not found at {path}")

    def load_pubmedqa(self) -> pd.DataFrame:
        """Load PubMedQA dataset."""
        path = self.data_dir / "pubmed" / "pubmedqa_labeled.parquet"
        df = self._safe_load_parquet(path)
        if df is not None:
            return df
        raise FileNotFoundError(f"PubMedQA not found at {path}")

    def load_medmcqa(self) -> pd.DataFrame:
        """Load MedMCQA dataset."""
        path = self.data_dir / "mediqa" / "medmcqa_train.parquet"
        df = self._safe_load_parquet(path)
        if df is not None:
            return df
        raise FileNotFoundError(f"MedMCQA not found at {path}")

    def load_healthcare_magic(self) -> pd.DataFrame:
        """Load HealthCareMagic dataset."""
        path = self.data_dir / "mediqa" / "healthcare_magic.parquet"
        df = self._safe_load_parquet(path)
        if df is not None:
            return df
        raise FileNotFoundError(f"HealthCareMagic not found at {path}")

    # ==================== New Standard Datasets ====================

    def load_medqa_usmle(self) -> pd.DataFrame:
        """Load MedQA USMLE dataset."""
        # Try train first, then test
        for filename in ["medqa_usmle_train.parquet", "medqa_usmle_test.parquet"]:
            path = self.data_dir / "medqa" / filename
            df = self._safe_load_parquet(path)
            if df is not None:
                return df
        raise FileNotFoundError("MedQA USMLE not found")

    def load_chatdoctor(self) -> pd.DataFrame:
        """Load ChatDoctor datasets."""
        dfs = []
        for filename in [
            "chatdoctor_icliniq.parquet",
            "chatdoctor_healthcaremagic.parquet",
        ]:
            path = self.data_dir / "chatdoctor" / filename
            df = self._safe_load_parquet(path)
            if df is not None:
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        raise FileNotFoundError("ChatDoctor datasets not found")

    def load_medical_meadow(self) -> pd.DataFrame:
        """Load all Medical Meadow datasets."""
        dfs = []
        meadow_dir = self.data_dir / "medical_meadow"

        if meadow_dir.exists():
            for parquet_file in meadow_dir.glob("*.parquet"):
                df = self._safe_load_parquet(parquet_file)
                if df is not None:
                    df["meadow_source"] = parquet_file.stem
                    dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        raise FileNotFoundError("Medical Meadow datasets not found")

    def load_additional_qa(self) -> pd.DataFrame:
        """Load additional consolidated QA datasets."""
        path = self.data_dir / "additional" / "lavita_medical_qa.parquet"
        df = self._safe_load_parquet(path)
        if df is not None:
            return df
        raise FileNotFoundError("Additional QA datasets not found")

    # ==================== Unified Loading ====================

    def load_all_qa_pairs(self) -> List[Dict]:
        """Load all QA pairs from all datasets."""
        qa_pairs = []

        # 1. Load MedQuAD
        try:
            df = self.load_medquad()
            for _, row in df.iterrows():
                qa_pairs.append(
                    {
                        "question": row.get("Question", row.get("question", "")),
                        "answer": row.get("Answer", row.get("answer", "")),
                        "source": "MedQuAD",
                    }
                )
            logger.info(f"  Loaded MedQuAD: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] MedQuAD: {e}")

        # 2. Load PubMedQA
        try:
            df = self.load_pubmedqa()
            for _, row in df.iterrows():
                context = row.get("context", {})
                if isinstance(context, dict):
                    context_text = " ".join(context.get("contexts", []))
                else:
                    context_text = str(context) if context else ""
                qa_pairs.append(
                    {
                        "question": row.get("question", ""),
                        "answer": row.get("long_answer", ""),
                        "context": context_text,
                        "source": "PubMedQA",
                    }
                )
            logger.info(f"  Loaded PubMedQA: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] PubMedQA: {e}")

        # 3. Load MedMCQA
        try:
            df = self.load_medmcqa()
            count = 0
            for _, row in df.iterrows():
                # Use explanation as the answer
                answer = row.get("exp")
                if not answer or pd.isna(answer):
                    continue

                # Construct full answer with correct option
                cop = row.get("cop")  # 1-based index
                options = {1: "opa", 2: "opb", 3: "opc", 4: "opd"}
                correct_opt = ""
                if cop in options:
                    correct_opt = row.get(options[cop], "")

                full_answer = f"Correct answer: {correct_opt}\n\nExplanation: {answer}"

                qa_pairs.append(
                    {
                        "question": row.get("question", ""),
                        "answer": full_answer,
                        "source": f"MedMCQA/{row.get('subject_name', 'General')}",
                    }
                )
                count += 1
            logger.info(f"  Loaded MedMCQA: {count:,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] MedMCQA: {e}")

        # 4. Load HealthCareMagic
        try:
            df = self.load_healthcare_magic()
            for _, row in df.iterrows():
                # Has instruction (question context), input (actual question), output (answer)
                question = row.get("input", row.get("instruction", ""))
                instruction = row.get("instruction", "")
                if instruction and question and instruction != question:
                    question = f"{instruction}\n\n{question}"

                qa_pairs.append(
                    {
                        "question": question,
                        "answer": row.get("output", ""),
                        "source": "HealthCareMagic",
                    }
                )
            logger.info(f"  Loaded HealthCareMagic: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] HealthCareMagic: {e}")

        # 5. Load MedQA USMLE
        try:
            df = self.load_medqa_usmle()
            for _, row in df.iterrows():
                question = row.get("question", row.get("sent1", ""))
                answer = row.get("answer", row.get("ending0", ""))

                # Some formats have options array
                options = row.get("options", [])
                answer_idx = row.get("answer_idx", row.get("label", -1))

                if options and isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    answer = options[answer_idx]

                if question and answer:
                    qa_pairs.append(
                        {
                            "question": question,
                            "answer": str(answer),
                            "source": "MedQA-USMLE",
                        }
                    )
            logger.info(f"  Loaded MedQA USMLE: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] MedQA USMLE: {e}")

        # 6. Load ChatDoctor
        try:
            df = self.load_chatdoctor()
            for _, row in df.iterrows():
                # ChatDoctor format: instruction/input/output or question/answer
                question = row.get("input", row.get("instruction", row.get("question", "")))
                answer = row.get("output", row.get("answer", ""))

                if question and answer:
                    qa_pairs.append(
                        {"question": question, "answer": answer, "source": "ChatDoctor"}
                    )
            logger.info(f"  Loaded ChatDoctor: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] ChatDoctor: {e}")

        # 7. Load Medical Meadow
        try:
            df = self.load_medical_meadow()
            for _, row in df.iterrows():
                # Medical Meadow format: instruction/input/output
                instruction = row.get("instruction", "")
                input_text = row.get("input", "")
                output_text = row.get("output", "")
                source = row.get("meadow_source", "MedicalMeadow")

                question = instruction
                if input_text:
                    question = f"{instruction}\n\n{input_text}" if instruction else input_text

                if question and output_text:
                    qa_pairs.append(
                        {
                            "question": question,
                            "answer": output_text,
                            "source": f"MedicalMeadow/{source}",
                        }
                    )
            logger.info(f"  Loaded Medical Meadow: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] Medical Meadow: {e}")

        # 8. Load Additional QA
        try:
            df = self.load_additional_qa()
            for _, row in df.iterrows():
                question = row.get("question", row.get("input", ""))
                answer = row.get("answer", row.get("output", ""))

                if question and answer:
                    qa_pairs.append(
                        {
                            "question": question,
                            "answer": answer,
                            "source": "LavitaMedicalQA",
                        }
                    )
            logger.info(f"  Loaded Additional QA: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"  [SKIP] Additional QA: {e}")

        logger.info(f"Total QA pairs loaded: {len(qa_pairs):,}")
        return qa_pairs

    def get_documents_for_knowledge_base(self) -> Generator[Dict, None, None]:
        """Yield documents for building knowledge base."""
        logger.info("Loading all QA pairs for knowledge base...")
        qa_pairs = self.load_all_qa_pairs()

        for qa in qa_pairs:
            # Combine question and answer as document
            content = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"
            if qa.get("context"):
                content = f"Context: {qa['context']}\n\n{content}"

            yield {
                "content": content,
                "source": qa["source"],
                "metadata": {"type": "qa_pair"},
            }

    def get_stats(self) -> Dict:
        """Get statistics about available datasets."""
        stats = {}

        datasets = [
            ("MedQuAD", self.load_medquad),
            ("PubMedQA", self.load_pubmedqa),
            ("MedMCQA", self.load_medmcqa),
            ("HealthCareMagic", self.load_healthcare_magic),
            ("MedQA-USMLE", self.load_medqa_usmle),
            ("ChatDoctor", self.load_chatdoctor),
            ("MedicalMeadow", self.load_medical_meadow),
            ("AdditionalQA", self.load_additional_qa),
        ]

        total = 0
        for name, loader in datasets:
            try:
                df = loader()
                stats[name] = len(df)
                total += len(df)
            except Exception:
                stats[name] = 0

        stats["total"] = total
        return stats
