from PyPDF2 import PdfReader
import re

reader = PdfReader("mysore_overview.pdf")
pages  = [p.extract_text() for p in reader.pages]

# Simple chunker: ~500 characters with 100-char overlap
def chunk_text(text, size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i+size]
        chunks.append(chunk)
        i += size - overlap
    return chunks

pdf_chunks = []
for pi, pg in enumerate(pages):
    for ci, chunk in enumerate(chunk_text(pg)):
        pdf_chunks.append({
            "id":   f"pdf_{pi}_{ci}",
            "text": chunk,
            "meta": {"source":"mysore_pdf", "page": pi, "chunk_id": ci}
        })
