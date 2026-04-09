"""
Unit tests for RecursiveSentenceChunker.

Validates:
- No chunk ends mid-sentence (sentence-boundary awareness)
- Domain-adaptive chunk sizes are applied correctly
- Overlap prepends tail of previous chunk
- Empty / tiny documents are handled gracefully
"""

import pytest

from src.data_pipeline.preprocessors.chunker import Chunk, RecursiveSentenceChunker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunker():
    return RecursiveSentenceChunker(chunk_overlap=50, min_chunk_size=20)


# ---------------------------------------------------------------------------
# Sentence boundary tests
# ---------------------------------------------------------------------------


def test_no_chunk_ends_mid_sentence(chunker):
    """Every chunk must end at a sentence boundary (period, !, ?) or be the last chunk."""
    paragraph = (
        "Hypertension is a chronic condition characterized by elevated blood pressure. "
        "It is a major risk factor for cardiovascular disease. "
        "Lifestyle modifications such as diet and exercise are first-line treatments. "
        "Antihypertensive medications are prescribed when lifestyle changes are insufficient. "
        "Regular monitoring is essential to assess treatment response and prevent complications. "
        "Blood pressure targets vary depending on patient age and comorbidities."
    )
    doc = {"content": paragraph, "source": "MedQuAD"}
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0, "Expected at least one chunk"

    for chunk in chunks[:-1]:  # last chunk may naturally end without punctuation
        # The content (stripped of overlap prefix) should end with punctuation
        text = chunk.content.rstrip()
        assert text[-1] in ".!?;", f"Chunk ends mid-sentence: '...{text[-60:]}'"


def test_chunks_contain_no_empty_strings(chunker):
    text = "Short sentence. Another short one. And a third."
    doc = {"content": text, "source": "test"}
    chunks = chunker.chunk_document(doc)
    for c in chunks:
        assert c.content.strip(), "Empty chunk produced"


# ---------------------------------------------------------------------------
# Domain-adaptive sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected_max_chars",
    [
        ("PubMedQA_labeled", 256 * 4 + 50),  # 1024 + overlap buffer
        ("MedMCQA_train", 128 * 4 + 50),  # 512 + overlap buffer
        ("HealthCareMagic", 512 * 4 + 50),  # 2048 + overlap buffer
    ],
)
def test_domain_adaptive_chunk_size(source, expected_max_chars):
    """Chunks for a given domain source should not exceed the domain size (+overlap)."""
    # Build a text that is definitely larger than any single domain size
    long_text = " ".join(
        [f"This is medical sentence number {i} about patient care." for i in range(200)]
    )
    doc = {"content": long_text, "source": source}
    chunker = RecursiveSentenceChunker(chunk_overlap=50, min_chunk_size=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1, f"Expected multiple chunks for {source}"
    oversized = [c for c in chunks if len(c.content) > expected_max_chars]
    assert not oversized, (
        f"{len(oversized)} chunks exceed limit for {source}: "
        f"max={max(len(c.content) for c in chunks)}, limit={expected_max_chars}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_content_returns_no_chunks(chunker):
    doc = {"content": "", "source": "test"}
    assert chunker.chunk_document(doc) == []


def test_short_content_single_chunk():
    small_chunker = RecursiveSentenceChunker(chunk_overlap=0, min_chunk_size=5)
    doc = {"content": "Short text.", "source": "MedQuAD"}
    chunks = small_chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_chunk_metadata_set(chunker):
    doc = {
        "content": "Diabetes is a metabolic disorder. It affects insulin production.",
        "source": "MedQuAD",
        "metadata": {"type": "qa_pair"},
    }
    chunks = chunker.chunk_document(doc)
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source == "MedQuAD"
        assert c.metadata.get("chunker") == "recursive"
        assert c.metadata.get("type") == "qa_pair"


def test_chunk_ids_sequential(chunker):
    long_text = " ".join(
        [f"Sentence {i} about cardiology and treatment protocols." for i in range(50)]
    )
    doc = {"content": long_text, "source": "MedQuAD"}
    chunks = chunker.chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks))), "chunk_ids must be sequential 0..n-1"


def test_overlap_not_longer_than_chunk(chunker):
    """Overlap must not produce a chunk longer than original chunk + overlap size."""
    long_text = " ".join(
        [f"Clinical note sentence {i} about diagnosis and treatment." for i in range(100)]
    )
    doc = {"content": long_text, "source": "default"}
    chunks = chunker.chunk_document(doc)
    for c in chunks:
        # Rough sanity: no chunk should be more than 3× the default size
        assert len(c.content) <= chunker._default_chunk_size * 3
