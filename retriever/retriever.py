import os
import uuid
import openai
import pinecone
import dotenv

# Load environment variables
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

# Constants
DATA_DIR = "./data"
CHUNK_SIZE = 500  # Characters, adjust as needed
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
        
        path = os.path.join(DATA_DIR, file_name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"Processing '{file_name}' with {len(chunks)} chunks...")

        for i in range(0, len(chunks), 10):  # Batch embed 10 at a time
            batch = chunks[i:i + 10]
            embeddings = embed_text_batch(batch)

            upserts = []
            for chunk_text, embedding in zip(batch, embeddings):
                vector_id = str(uuid.uuid4())
                metadata = {
                    "text": chunk_text,
                    "source": file_name,
                    "chunk_id": i
                }
                upserts.append((vector_id, embedding, metadata))

            index.upsert(vectors=upserts)

        print(f"✔ Uploaded '{file_name}' to Pinecone.")

if __name__ == "__main__":
    ingest_documents()
