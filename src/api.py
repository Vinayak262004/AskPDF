

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
    load_faiss_index,   
    search,             
)
from src.rag import answer_with_llm   


app = FastAPI()


@app.get("/")
def root():
    return {"status": "API is running!"}




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


    save_faiss_index(index)  

    return {
        "message": "PDF processed and FAISS index created!",
        "num_chunks": len(chunks),
    }




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
        
        if isinstance(ctx, dict):
            idx = ctx.get("index")
            dist = ctx.get("distance")
            text = ctx.get("text") or ctx.get("page_content") or ""
        
        elif hasattr(ctx, "page_content"):
            idx = getattr(ctx, "index", None)
            dist = getattr(ctx, "distance", None)
            text = ctx.page_content
        
        elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
            
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
                "text": text[:800],  
            }
        )
    return formatted

@app.post("/ask")
def ask_json(payload: Question):
    question = payload.question
    
   
    answer, context = answer_with_llm(question, k=3)

    
    return {
        "question": question,
        "answer": answer
    }




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
