"""
LangChain-compatible retriever wrapper for HybridRetriever.

Wraps the existing HybridRetriever to work with LangChain's LCEL pipelines.
"""

# Import the existing retriever
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


from src.retrieval.hybrid_retriever import RetrievedDocument


class LangChainHybridRetriever(BaseRetriever):
    """
    LangChain-compatible wrapper for HybridRetriever.

    This wrapper allows using the existing HybridRetriever with LangChain's
    LCEL for declarative pipeline composition.

    Example:
        retriever = LangChainHybridRetriever(hybrid_retriever=my_retriever)
        docs = retriever.invoke("What is diabetes?")
    """

    hybrid_retriever: Any = Field(description="The underlying HybridRetriever instance")
    k: int = Field(default=5, description="Number of documents to retrieve")
    use_hybrid: bool = Field(default=True, description="Whether to use hybrid retrieval")
    use_reranking: bool = Field(default=True, description="Whether to apply reranking")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any):
        """
        Initialize wrapper.

        Accepts both `hybrid_retriever=` and legacy `retriever=` argument
        for backward compatibility with older tests/callers.
        """
        if "retriever" in kwargs and "hybrid_retriever" not in kwargs:
            kwargs["hybrid_retriever"] = kwargs.pop("retriever")
        super().__init__(**kwargs)

    @property
    def _retriever(self) -> Any:
        """Legacy alias expected by older tests/callers."""
        return self.hybrid_retriever

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            List of LangChain Document objects
        """
        # Use the underlying retriever
        retrieved_docs: List[RetrievedDocument] = self.hybrid_retriever.retrieve(
            query=query,
            k=self.k,
            use_hybrid=self.use_hybrid,
            use_reranking=self.use_reranking,
        )

        # Convert to LangChain Document format
        documents = []
        for doc in retrieved_docs:
            langchain_doc = Document(
                page_content=doc.content,
                metadata={"source": doc.source, "score": doc.score, **doc.metadata},
            )
            documents.append(langchain_doc)

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version - delegates to sync for now."""
        return self._get_relevant_documents(query, run_manager=run_manager)

    def retrieve_with_context(
        self, query: str, max_context_length: int = 2000
    ) -> tuple[List[Document], str]:
        """
        Retrieve documents and build context string.

        Args:
            query: Search query
            max_context_length: Maximum context length

        Returns:
            Tuple of (documents, context_string)
        """
        docs, context = self.hybrid_retriever.retrieve_with_context(
            query=query,
            k=self.k,
            max_context_length=max_context_length,
            use_reranking=self.use_reranking,
        )

        # Convert to LangChain Documents
        langchain_docs = []
        for doc in docs:
            langchain_doc = Document(
                page_content=doc.content,
                metadata={"source": doc.source, "score": doc.score, **doc.metadata},
            )
            langchain_docs.append(langchain_doc)

        return langchain_docs, context


def format_docs_as_context(docs: List[Document], max_length: int = 2000) -> str:
    """
    Format LangChain documents as a context string for prompts.

    Args:
        docs: List of LangChain Documents
        max_length: Maximum context length

    Returns:
        Formatted context string
    """
    context_parts = []
    total_length = 0

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("score", 0.0)

        entry = f"[Source: {source} (relevance: {score:.2f})]\n{doc.page_content}"

        if total_length + len(entry) > max_length:
            break

        context_parts.append(entry)
        total_length += len(entry)

    return "\n\n---\n\n".join(context_parts)


def docs_to_retrieved_documents(docs: List[Document]) -> List[RetrievedDocument]:
    """
    Convert LangChain Documents back to RetrievedDocument format.

    Useful for compatibility with existing XAI components.
    """
    return [
        RetrievedDocument(
            content=doc.page_content,
            source=doc.metadata.get("source", "Unknown"),
            score=doc.metadata.get("score", 0.0),
            metadata={k: v for k, v in doc.metadata.items() if k not in ["source", "score"]},
        )
        for doc in docs
    ]
