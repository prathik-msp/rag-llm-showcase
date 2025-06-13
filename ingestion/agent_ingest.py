import os
import uuid
import openai
from pinecone import Pinecone
import dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

EMBED_MODEL = "text-embedding-3-small"

def chunk_text(text: str, chunk_size: int = 500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_text_batch(text_batch: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text_batch
    )
    return [item.embedding for item in response.data]

def summarize_text(text: str) -> str:
    summary_prompt = f"Summarize the following document:\n\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def ingest_file(filename: str, content: str):
    summary = summarize_text(content)
    chunks = chunk_text(content)

    for i in range(0, len(chunks), 10):
        batch = chunks[i:i + 10]
        embeddings = embed_text_batch(batch)

        upserts = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            vector_id = str(uuid.uuid4())
            metadata = {
                "text": chunk,
                "source": filename,
                "summary": summary,
                "chunk_id": i + j
            }
            upserts.append((vector_id, embedding, metadata))

        index.upsert(vectors=upserts)
