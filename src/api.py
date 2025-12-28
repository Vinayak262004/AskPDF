# src/api.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion import extract_text_from_pdf, chunk_text_tokens
from src.vectorstore import (
    embed_texts,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,   # <-- make sure this exists
    search,             # <-- search(index, query, k) or similar
)
from src.rag import answer_with_llm   # this will use `search` + LLM


app = FastAPI()


@app.get("/")
def root():
    return {"status": "API is running!"}


# --------- 1) /upload-pdf ----------
# src/api.py

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = Path("uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    pages = extract_text_from_pdf(pdf_path)
    text = "\n\n".join(pages)
    chunks = chunk_text_tokens(text, max_tokens=300, overlap=80)

    Path("data").mkdir(exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    vectors = embed_texts(chunks)
    index = build_faiss_index(vectors)

    # ✅ this now matches the updated vectorstore.save_faiss_index
    save_faiss_index(index)  # or save_faiss_index(index, "data/index.faiss")

    return {
        "message": "PDF processed and FAISS index created!",
        "num_chunks": len(chunks),
    }



# --------- 2) /ask ----------

class Question(BaseModel):
    question: str


def format_context(context):
    """
    Normalize context into a list of dicts:
    [
        {"rank": 1, "index": 0, "distance": 0.12, "text": "..."},
        ...
    ]
    """
    formatted = []
    for rank, ctx in enumerate(context, start=1):
        # Handle several possible shapes to avoid crashes:
        # 1) ctx is already a dict
        if isinstance(ctx, dict):
            idx = ctx.get("index")
            dist = ctx.get("distance")
            text = ctx.get("text") or ctx.get("page_content") or ""
        # 2) ctx is a LangChain Document
        elif hasattr(ctx, "page_content"):
            idx = getattr(ctx, "index", None)
            dist = getattr(ctx, "distance", None)
            text = ctx.page_content
        # 3) ctx is a tuple like (doc, score) or (score, text)
        elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
            # very defensive: try to guess
            first, second = ctx[0], ctx[1]
            idx = None
            dist = float(second) if isinstance(second, (float, int)) else None
            text = getattr(first, "page_content", None) or str(first)
        else:
            idx, dist, text = None, None, str(ctx)

        formatted.append(
            {
                "rank": rank,
                "index": idx,
                "distance": float(dist) if dist is not None else None,
                "text": text[:800],   # trim for UI
            }
        )
    return formatted

@app.post("/ask")
def ask_json(payload: Question):
    question = payload.question
    
    # RAG pipeline still runs, but we ignore context for UI
    answer, context = answer_with_llm(question, k=3)

    # return only clean answer
    return {
        "question": question,
        "answer": answer
    }


# --------- CORS ----------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # frontend on localhost:8501
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
