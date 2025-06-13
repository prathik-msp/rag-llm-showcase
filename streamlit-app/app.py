import streamlit as st
import requests

# Change this to 'backend:8000' if you're using Docker Compose
BACKEND_URL = "http://backend:8000"

st.title("Sai's Agentic RAG Showcase for LAM")

# 3 tabs
tab1, tab2, tab3 = st.tabs(["Regular RAG", "Agentic RAG", "Architecture Diagram"])

# Tab 1: Regular RAG
with tab1:
    query = st.text_input("Ask a question (Regular RAG)", key="regular_rag_input")
    if st.button("Ask (RAG)", key="regular_rag_button"):
        try:
            response = requests.post(f"{BACKEND_URL}/ask", json={"query": query})
            result = response.json()

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            for i, source in enumerate(result["sources"]):
                with st.expander(f"Source {i+1} — File: {source['source']} — Score: {source['score']:.3f}"):
                    st.write(source["text"])

            st.subheader("Tokens Used")
            st.write(result["tokens_used"])

        except Exception as e:
            st.error(f"Error occurred: {e}")

# Tab 2: Agentic RAG
with tab2:
    st.header("Upload files for Agentic RAG")

    uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            contents = file.read().decode("utf-8")
            response = requests.post(
                "http://backend:8000/agent_ingest",
                json={"filename": file.name, "content": contents}
            )
            if response.ok:
                st.success(f"Ingested {file.name}")
            else:
                st.error(f"Failed to ingest {file.name}")

    st.header("Ask Agentic RAG")
    agent_query = st.text_input("Your agentic question:")
    if st.button("Ask (Agentic RAG)"):
        response = requests.post("http://backend:8000/agentic_ask", json={"query": agent_query})
        if response.ok:
            result = response.json()
            st.subheader("Answer")
            st.write(result["answer"]["answer"])

            st.subheader("Sources")
            sources = result["answer"]["sources"]
            if isinstance(sources, dict):
                sources = list(sources.values())

            for i, source in enumerate(sources):
                with st.expander(f"Source {i+1} — File: {source['source']} — Score: {source['score']:.3f}"):
                    st.write(source["text"])
            
            st.subheader("Tokens Used")
            st.write(result["tokens_used"])

          
        else:
            st.error("Agentic query failed.")


# Tab 3: Architecture Diagram
with tab3:
    st.subheader("Agentic RAG Architecture Diagram")

    architecture = """
### RAG Pipeline Comparison

#### 🔷 Regular RAG Workflow

Document Upload -> text chunking / Embedding -> vector Database (pinecone) <br>

──────────── Query Time ────────────<br>
User Query <br>↓<br>
Embed Query <br>↓<br>
Vector Search (Pinecone Top-K retrieval) <br>↓<br>
LLM Prompt Engineering <br>↓<br>
LLM Generates Answer


---

#### 🔷 Agentic RAG Workflow

Document Upload -> Text Chunking (recursive with overlap) -> Metadata Extraction (Summaries, Entities, Keywords) -> Embeddings (OpenAI text-embedding-3-small) -> Vector Storage (Pinecone with Metadata)

──────────── Query Time ────────────<br>
User Query <br>↓<br>
LLM Query Decomposition <br>↓<br>
Sub-Queries <br>↓<br>
For Each Sub-Query:<br>
→ Embed Sub-Query<br>
→ Vector Search (Pinecone Retrieval)<br>
→ Context → LLM Sub-Answer <br>↓<br>
Sub-Answers Collected <br>↓<br>
LLM Final Synthesis <br>↓<br>
Final Answer
    """

    st.markdown(architecture, unsafe_allow_html=True)
