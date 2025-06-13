import os
import openai
from pinecone import Pinecone
import dotenv
import json

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

EMBED_MODEL = "text-embedding-3-small"

def decompose_query(query: str):
    prompt = f"Break this query into smaller sub-questions:\n\n{query}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    subs = response.choices[0].message.content.strip().split("\n")
    return [s.strip() for s in subs if s.strip()]

def embed_query(text: str):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def retrieve_for_subquery(subquery: str):
    embedding = embed_query(subquery)
    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    return [
        {
            "text": match['metadata'].get('text', ''),
            "score": match['score'],
            "source": match['metadata'].get('source', 'unknown')
        }
        for match in result.get('matches', [])
    ]

def answer_subquery(subquery: str, context_chunks):
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    prompt = f"""Use the following context to answer the question:
    
Context:
{context}

Question: {subquery}
Answer:"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def agentic_rag(query: str):
    sub_questions = decompose_query(query)
    sub_answers = []
    all_sources = []

    for sub in sub_questions:
        chunks = retrieve_for_subquery(sub)
        answer = answer_subquery(sub, chunks)
        sub_answers.append(f"Sub-question: {sub}\nAnswer: {answer}\n")
        all_sources.extend(chunks)

    synthesis_prompt = "Synthesize a full answer from these sub-answers:\n\n" + "\n\n".join(sub_answers)

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": synthesis_prompt}]
    )
    final_answer = response.choices[0].message.content.strip()
    clean_sources = json.loads(json.dumps(all_sources))

    return {
        "answer": final_answer,
        "sources": list(all_sources)
    }
