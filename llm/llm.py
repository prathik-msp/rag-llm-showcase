import os
import openai
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

def build_prompt(context_chunks: List[Dict[str, str]], query: str) -> str:
    context_text = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}" for chunk in context_chunks
    )

    return f"""
You are an intelligent assistant for internal company documentation. Use ONLY the information from the context below to answer the question. If the answer is not in the context, respond with "I don't know".

Context:
---------
{context_text}
---------

Question: {query}
Answer:
"""

def generate_answer(chunks: List[Dict[str, str]], query: str) -> Dict:
    prompt = build_prompt(chunks, query)

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "tokens_used": response.usage.total_tokens
    }
