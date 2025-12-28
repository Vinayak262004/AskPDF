# src/rag.py

import os
from typing import List, Tuple
from pathlib import Path

from dotenv import load_dotenv

from src.vectorstore import search

load_dotenv()

try:
    from openai import OpenAI

    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _client = None  


def get_relevant_context(question: str, k: int = 3) -> Tuple[str, List[dict]]:
    
    results = search(question, k=k)

    
    results = sorted(results, key=lambda r: r["distance"])

    
    parts = []
    for r in results:
        parts.append(f"[CHUNK {r['index']} | dist={r['distance']:.3f}]\n{r['text']}\n")

    context_str = "\n\n".join(parts)
    return context_str, results


def build_prompt(question: str, context: str) -> str:
    
    prompt = f"""
You are a helpful assistant that answers questions about a document.

Use ONLY the information in the CONTEXT below. If the answer is not in the context,
say "I don't know from this document" and do NOT hallucinate.

CONTEXT:
{context}

QUESTION:
{question}

Answer in 3–6 sentences, concise and clear.
"""
    return prompt.strip()


def answer_with_llm(question: str, k: int = 3) -> Tuple[str, str]:
    
    context, _ = get_relevant_context(question, k=k)
    prompt = build_prompt(question, context)

    
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if _client is None or not api_key:
        fallback_answer = (
            "OpenAI API key not configured or openai library not available.\n\n"
            "Here is the retrieved context instead:\n\n"
            + context
        )
        return fallback_answer, context

   
    completion = _client.chat.completions.create(
        model="gpt-4.1-mini",  # you can change model if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant for question answering over documents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer_text = completion.choices[0].message.content
    return answer_text, context
