
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.json")
FAISS_PATH = Path("data/faiss.index")


_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")




def embed_texts(texts: list[str]) -> np.ndarray:
   
    emb = _model.encode(texts, convert_to_numpy=True)
    return emb.astype("float32")


def embed_query(query: str) -> np.ndarray:
  
    emb = _model.encode([query], convert_to_numpy=True)
    return emb.astype("float32")




def build_faiss_index(vectors: np.ndarray):
   
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

DEFAULT_INDEX_PATH = Path("data/index.faiss")
def save_faiss_index(index, index_path: Path = DEFAULT_INDEX_PATH):

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
def load_faiss_index():
 
    return faiss.read_index(str(FAISS_PATH))




def save_chunks(chunks: list[str]) -> None:
    
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_chunks() -> list[str]:
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)



def search(query: str, k: int = 3):
   
    index = load_faiss_index()
    q_vec = embed_query(query) 
    distances, indices = index.search(q_vec, k)  

    chunks = load_chunks()
    idxs = indices[0]
    dists = distances[0]

    results = []
    for i, d in zip(idxs, dists):
        idx = int(i)
        if idx >= len(chunks):
            continue
        chunk_text = chunks[idx]

        results.append(
            {"index": int(i), "distance": float(d), "text": chunk_text}
        )

    return results
