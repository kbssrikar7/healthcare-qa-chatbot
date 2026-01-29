#!/usr/bin/env python3
"""
Test the complete QA pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("🧪 Testing Healthcare QA Pipeline\n")
    print("=" * 50)
    
    # Initialize components
    print("1️⃣ Initializing components...")
    
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
    
    # Test questions
    test_questions = [
        "What are the symptoms of diabetes?",
        "How can I lower my blood pressure?",
        "What causes headaches?"
    ]
    
    print("\n2️⃣ Testing questions...\n")
    
    for q in test_questions:
        print(f"Question: {q}")
        print("-" * 40)
        
        try:
            response = pipeline.answer(q, num_documents=3, include_explanation=True)
            print(f"Answer: {response.answer[:300]}...")
            print(f"Confidence: {response.confidence['level']} ({response.confidence['score']:.2f})")
            print(f"Sources: {len(response.sources)}")
            print()
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("=" * 50)
    print("✅ Pipeline test complete!")

if __name__ == "__main__":
    main()
