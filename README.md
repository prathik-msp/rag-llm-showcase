# rag-llm-showcase

Agentic RAG Architecture Diagram
RAG Pipeline Comparison
🔷 Regular RAG Workflow
Document Upload → text chunking / Embedding → vector Database (pinecone)

──────────── Query Time ────────────
User Query
↓
Embed Query
↓
Vector Search (Pinecone Top-K retrieval)
↓
LLM Prompt Engineering
↓
LLM Generates Answer



🔷 Agentic RAG Workflow
Document Upload → Text Chunking (recursive with overlap) → Metadata Extraction (Summaries, Entities, Keywords) → Embeddings (OpenAI text-embedding-3-small) → Vector Storage (Pinecone with Metadata)

──────────── Query Time ────────────
User Query
↓
LLM Query Decomposition
↓
Sub-Queries
↓
For Each Sub-Query:
→ Embed Sub-Query
→ Vector Search (Pinecone Retrieval)
→ Context → LLM Sub-Answer
↓
Sub-Answers Collected
↓
LLM Final Synthesis
↓
Final Answer