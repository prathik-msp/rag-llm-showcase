from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(req: AskRequest):
    return {
        "answer": "This is a dummy response.",
        "sources": [],
        "tokens_used": 0
    }
