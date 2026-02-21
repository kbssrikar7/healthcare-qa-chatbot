#!/usr/bin/env python3
"""
Evaluate the Healthcare QA pipeline on benchmarks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from tqdm import tqdm
from datetime import datetime
import numpy as np

def evaluate_retrieval(pipeline, test_questions):
    """Evaluate retrieval performance."""
    results = []
    
    for q in tqdm(test_questions, desc="Retrieval eval"):
        retrieved = pipeline.retriever.retrieve(q["question"], k=10)
        
        # Check if relevant content is retrieved
        relevant_found = any(
            q.get("expected_topic", "").lower() in r.content.lower()
            for r in retrieved
        )
        
        results.append({
            "question": q["question"],
            "retrieved_count": len(retrieved),
            "relevant_found": relevant_found,
            "top_score": retrieved[0].score if retrieved else 0
        })
    
    # Calculate metrics
    recall = sum(1 for r in results if r["relevant_found"]) / len(results) if results else 0
    avg_score = np.mean([r["top_score"] for r in results]) if results else 0
    
    return {
        "recall@10": recall,
        "avg_retrieval_score": avg_score,
        "total_questions": len(results)
    }

def evaluate_generation(pipeline, test_questions):
    """Evaluate generation quality."""
    results = []
    
    for q in tqdm(test_questions, desc="Generation eval"):
        try:
            response = pipeline.answer(
                q["question"],
                num_documents=5,
                include_explanation=True
            )
            
            results.append({
                "question": q["question"],
                "answer_length": len(response.answer),
                "confidence_score": response.confidence["score"],
                "confidence_level": response.confidence["level"],
                "num_sources": len(response.sources),
                "num_attributions": len(response.attributions)
            })
        except Exception as e:
            results.append({
                "question": q["question"],
                "error": str(e)
            })
    
    # Calculate metrics
    successful = [r for r in results if "error" not in r]
    
    if successful:
        avg_confidence = np.mean([r["confidence_score"] for r in successful])
        avg_sources = np.mean([r["num_sources"] for r in successful])
        high_confidence = sum(1 for r in successful if r["confidence_level"] == "high") / len(successful)
    else:
        avg_confidence = 0
        avg_sources = 0
        high_confidence = 0
    
    return {
        "avg_confidence": avg_confidence,
        "avg_sources": avg_sources,
        "high_confidence_rate": high_confidence,
        "success_rate": len(successful) / len(results) if results else 0,
        "total_questions": len(results)
    }

def evaluate_xai(pipeline, test_questions):
    """Evaluate XAI components."""
    results = []
    
    for q in tqdm(test_questions, desc="XAI eval"):
        try:
            response = pipeline.answer(
                q["question"],
                num_documents=5,
                include_explanation=True
            )
            
            # Check attribution coverage
            supported = sum(1 for a in response.attributions if a.get("source") != "Unsupported")
            coverage = supported / len(response.attributions) if response.attributions else 0
            
            results.append({
                "question": q["question"],
                "attribution_coverage": coverage,
                "confidence_provided": response.confidence["score"] > 0,
                "explanation_provided": bool(response.confidence.get("explanation"))
            })
        except Exception as e:
            results.append({
                "question": q["question"],
                "error": str(e)
            })
    
    # Calculate metrics
    successful = [r for r in results if "error" not in r]
    
    if successful:
        avg_coverage = np.mean([r["attribution_coverage"] for r in successful])
        confidence_rate = sum(1 for r in successful if r["confidence_provided"]) / len(successful)
    else:
        avg_coverage = 0
        confidence_rate = 0
    
    return {
        "avg_attribution_coverage": avg_coverage,
        "confidence_provision_rate": confidence_rate,
        "total_questions": len(results)
    }

def main():
    print("📊 Healthcare QA Pipeline Evaluation\n")
    print("=" * 50)
    
    # Sample test questions
    test_questions = [
        {"question": "What are the symptoms of diabetes?", "expected_topic": "diabetes"},
        {"question": "How is high blood pressure treated?", "expected_topic": "blood pressure"},
        {"question": "What causes migraines?", "expected_topic": "migraine"},
        {"question": "What is asthma?", "expected_topic": "asthma"},
        {"question": "How can I prevent heart disease?", "expected_topic": "heart"}
    ]
    
    try:
        # Initialize pipeline
        print("1️⃣ Initializing pipeline...")
        from src.embeddings.embedding_models import MedicalEmbedder
        from src.embeddings.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.generation.llm_wrapper import MedicalLLM
        from src.generation.prompt_manager import MedicalPromptManager
        from src.xai.confidence_scorer import ConfidenceScorer
        from src.xai.source_attribution import SourceAttributor
        from src.pipeline.qa_pipeline import HealthcareQAPipeline
        
        embedder = MedicalEmbedder(model_name="all-minilm")
        vector_store = VectorStore(
            collection_name="medical_knowledge",
            persist_directory="data/knowledge_base"
        )
        retriever = HybridRetriever(embedder, vector_store)
        llm = MedicalLLM(model_name="tinyllama", load_in_4bit=False)
        prompt_manager = MedicalPromptManager()
        confidence_scorer = ConfidenceScorer()
        source_attributor = SourceAttributor()
        
        pipeline = HealthcareQAPipeline(
            retriever=retriever,
            llm=llm,
            prompt_manager=prompt_manager,
            confidence_scorer=confidence_scorer,
            source_attributor=source_attributor
        )
        print("   ✅ Pipeline initialized")
        
        # Run evaluations
        print("\n2️⃣ Running evaluations...")
        
        retrieval_results = evaluate_retrieval(pipeline, test_questions)
        print(f"   Retrieval: Recall@10 = {retrieval_results['recall@10']:.2%}")
        
        generation_results = evaluate_generation(pipeline, test_questions)
        print(f"   Generation: Avg Confidence = {generation_results['avg_confidence']:.2%}")
        
        xai_results = evaluate_xai(pipeline, test_questions)
        print(f"   XAI: Attribution Coverage = {xai_results['avg_attribution_coverage']:.2%}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📈 Evaluation Summary")
        print("=" * 50)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "retrieval": retrieval_results,
            "generation": generation_results,
            "xai": xai_results
        }
        
        print(json.dumps(results, indent=2))
        
        # Save results
        output_path = Path("evaluation/results/evaluation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
