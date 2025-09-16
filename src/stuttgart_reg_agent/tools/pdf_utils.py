# src/stuttgart_reg_agent/tools/pdf_utils.py
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from PDF. Uses PyMuPDF for normal PDFs and OCR for scanned PDFs.
    """
    text = ""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
            else:
                # OCR fallback for scanned pages
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
    return text
