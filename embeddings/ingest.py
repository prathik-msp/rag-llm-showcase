import os
import uuid
import openai
from pinecone import Pinecone
import dotenv

# === Load .env secrets ===
dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX_NAME")

# === Connect to Pinecone ===
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# === Config ===
DATA_DIR = "./data"
CHUNK_SIZE = 500  # characters
EMBED_MODEL = "text-embedding-3-small"

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_text_batch(text_batch: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text_batch
    )
    return [item.embedding for item in response.data]

def ingest_documents():
    for file_name in os.listdir(DATA_DIR):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(DATA_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"📄 {file_name}: {len(chunks)} chunks")

        for i in range(0, len(chunks), 10):  # Batch size = 10
            batch = chunks[i:i + 10]
            embeddings = embed_text_batch(batch)

            upserts = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                vector_id = str(uuid.uuid4())
                metadata = {
                    "text": chunk,
                    "source": file_name,
                    "chunk_id": i + j
                }
                upserts.append((vector_id, embedding, metadata))


            index.upsert(vectors=upserts)

        print(f"✅ Uploaded '{file_name}' to Pinecone.\n")

if __name__ == "__main__":
    ingest_documents()
