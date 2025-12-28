
from pathlib import Path
import sys

from src.ingestion import extract_text_from_pdf, chunk_text_tokens
from src.vectorstore import (
    embed_texts,
    build_faiss_index,
    save_faiss_index,
    save_chunks,
)


def main(argv):
    if len(argv) < 2:
        print("Usage: python -m src.loader file.pdf")
        return 1

    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 1

  
    pages = extract_text_from_pdf(path)
    print(f"Loaded {len(pages)} pages from {path.name}.")

   
    text = "\n\n".join(pages)
    chunks = chunk_text_tokens(text, max_tokens=300, overlap=80)

    print(f"Created {len(chunks)} token chunks.")
    print("--- SAMPLE CHUNK ---")
    if chunks:
        print(chunks[0][:1000])
    else:
        print("(No chunks created; check PDF content.)")


    save_chunks(chunks)
    print("Saved chunks to data/chunks.json")


    print("Embedding chunks...")
    vectors = embed_texts(chunks)
    print(f"Embeddings shape: {vectors.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(vectors)


    save_faiss_index(index)
    print("FAISS index created and saved as data/faiss.index")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
