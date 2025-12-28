"""
Day 2: Robust PDF Ingestion + Token-based Chunking

Provides:
- extract_text_from_pdf(): multi-method extraction
- detect_scanned_pdf(): checks if OCR is needed
- ocr_extract(): performs OCR
- chunk_text_tokens(): token-based chunking using HF tokenizer
"""

from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract
import pytesseract
from PIL import Image
import io
from transformers import AutoTokenizer


# -----------------------------
#   PDF SCANNED DETECTION
# -----------------------------
def detect_scanned_pdf(path: Path) -> bool:
    """
    Heuristic: if pdfplumber can't detect text on first page, assume scanned.
    """
    try:
        with pdfplumber.open(path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
            return len(text.strip()) == 0
    except Exception:
        return False


# -----------------------------
#   OCR EXTRACTION
# -----------------------------
def ocr_extract(path: Path):
    """
    Converts each PDF page to an image then OCRs it.
    Slower but works for scanned PDFs.
    """
    import pdfplumber  # re-import inside if needed

    pages = []
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            img = pg.to_image(resolution=300).original
            text = pytesseract.image_to_string(img)
            pages.append(text)
    return pages


# -----------------------------
#   GENERAL EXTRACTION LOGIC
# -----------------------------
def extract_text_from_pdf(path: Path):
    """
    Multi-step extraction:
    1. Try pdfplumber
    2. Fallback to PyPDF2
    3. Fallback to pdfminer
    4. OCR scanned PDFs
    """
    path = Path(path)

    # 0. Check scanned PDF
    if detect_scanned_pdf(path):
        print("[INFO] Scanned PDF detected — using OCR...")
        return ocr_extract(path)

    # 1. pdfplumber
    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        if any(pages):
            return pages
    except:
        pass

    # 2. PyPDF2
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        if any(pages):
            return pages
    except:
        pass

    # 3. pdfminer
    try:
        text = pdfminer_extract(str(path))
        if text:
            return [text]
    except:
        pass

    return [""]


# -----------------------------
#   TOKEN-BASED CHUNKING
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text_tokens(text: str, max_tokens=300, overlap=80):
    """
    Token-based chunking for embeddings.
    """
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    N = len(tokens)

    while start < N:
        end = start + max_tokens
        chunk_ids = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)
        start = max(0, end - overlap)

    return chunks
