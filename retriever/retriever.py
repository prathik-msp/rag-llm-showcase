import os
import openai
from pinecone import Pinecone
import dotenv

dotenv.load_dotenv()

# Load env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
openai.api_key = OPENAI_API_KEY

EMBED_MODEL = "text-embedding-3-small"

def embed_query(text: str):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def retrieve_top_k(query: str, top_k: int = 5):
    query_embedding = embed_query(query)

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "text": match['metadata'].get('text', ''),
            "score": match['score'],
            "source": match['metadata'].get('source', 'unknown')
        }
        for match in result.get('matches', [])
    ]
