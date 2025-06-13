from fastapi import FastAPI
from pydantic import BaseModel
from retriever.retriever import retrieve_top_k
from llm.llm import generate_answer
from ingestion.agent_ingest import ingest_file
from agent.agentic_rag import agentic_rag

app = FastAPI()

class AskRequest(BaseModel):
    query: str

class UploadRequest(BaseModel):
    filename: str
    content: str

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(req: AskRequest):
    chunks = retrieve_top_k(req.query)
    result = generate_answer(chunks, req.query)
    return {
        "answer": result["answer"],
        "sources": chunks,
        "tokens_used": result["tokens_used"]
    }

@app.post("/agent_ingest")
def agent_ingest(req: UploadRequest):
    ingest_file(req.filename, req.content)
    return {"status": "ingested", "file": req.filename}

@app.post("/agentic_ask")
async def agentic_ask(request: QueryRequest):
    final_answer = agentic_rag(request.query)
    return {
        "answer": final_answer
    }
