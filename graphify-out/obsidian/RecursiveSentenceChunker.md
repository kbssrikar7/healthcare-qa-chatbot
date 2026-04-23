---
source_file: "/home/kbs/Documents/final_project/src/data_pipeline/preprocessors/chunker.py"
type: "code"
community: "Community 0"
location: "L217"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Community_0
---

# RecursiveSentenceChunker

## Connections
- [[.__init__()_57]] - `method` [EXTRACTED]
- [[._add_overlap()_1]] - `method` [EXTRACTED]
- [[._merge()]] - `method` [EXTRACTED]
- [[._size_for_source()]] - `method` [EXTRACTED]
- [[._split()]] - `method` [EXTRACTED]
- [[.chunk_document()_1]] - `method` [EXTRACTED]
- [[.chunk_documents()_1]] - `method` [EXTRACTED]
- [[Chunks for a given domain source should not exceed the domain size (+overlap).]] - `uses` [INFERRED]
- [[Every chunk must end at a sentence boundary (period, !, ) or be the last chunk.]] - `uses` [INFERRED]
- [[Overlap must not produce a chunk longer than original chunk + overlap size.]] - `uses` [INFERRED]
- [[Recursive text splitter that respects sentence boundaries.      Algorithm     --]] - `rationale_for` [EXTRACTED]
- [[Unit tests for RecursiveSentenceChunker.  Validates - No chunk ends mid-sentenc]] - `uses` [INFERRED]
- [[chunker()_1]] - `calls` [INFERRED]
- [[chunker.py]] - `contains` [EXTRACTED]
- [[main()_27]] - `calls` [INFERRED]
- [[test_domain_adaptive_chunk_size()]] - `calls` [INFERRED]
- [[test_short_content_single_chunk()]] - `calls` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/Community_0