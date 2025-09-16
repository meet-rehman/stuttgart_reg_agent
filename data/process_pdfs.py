# process_pdfs.py
import os, json, hashlib, argparse
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from tqdm import tqdm
import pandas as pd
import time

def file_checksum(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def extract_text_selectable(path, min_chars=200):
    try:
        doc = fitz.open(path)
        text_parts = []
        for page in doc:
            page_text = page.get_text("text") or ""
            text_parts.append(page_text)
            if sum(len(p) for p in text_parts) > min_chars:
                return "\n\n".join(text_parts)
        return "\n\n".join(text_parts)
    except Exception as e:
        return ""

def ocr_pdf(path, dpi=300, lang='deu'):
    pages = convert_from_path(path, dpi=dpi)
    texts = []
    for img in pages:
        # optional: convert to grayscale to help OCR
        if img.mode != "RGB":
            img = img.convert("RGB")
        txt = pytesseract.image_to_string(img, lang=lang, config='--psm 3')
        texts.append(txt)
    return "\n\n".join(texts)

def process_file(path, out_text_dir, out_meta_dir, ocr_needed_dir, min_chars=200):
    path = Path(path)
    meta = {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "checksum": file_checksum(path),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pages": None,
        "ocr_used": False,
        "text_path": None,
        "meta_path": None,
        "error": None
    }
    try:
        text = extract_text_selectable(str(path), min_chars=min_chars)
        if not text or len(text.strip()) < min_chars:
            # Needs OCR
            meta['ocr_used'] = True
            try:
                text = ocr_pdf(str(path), dpi=300, lang='deu')
            except Exception as e:
                meta['error'] = f"OCR error: {e}"
        # Save text
        text_fname = out_text_dir / (path.stem + ".txt")
        text_fname.parent.mkdir(parents=True, exist_ok=True)
        with open(text_fname, "w", encoding="utf-8") as f:
            f.write(text)
        meta['text_path'] = str(text_fname)
        # Move original to ocr_needed if OCR was used (optional)
        if meta['ocr_used']:
            dest = ocr_needed_dir / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                path.replace(dest)  # move file
        # Save meta
        meta_fname = out_meta_dir / (path.stem + ".meta.json")
        with open(meta_fname, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        meta['meta_path'] = str(meta_fname)
        return meta
    except Exception as e:
        meta['error'] = str(e)
        return meta

def main(input_dir="raw", out_text_dir="processed/texts", out_meta_dir="processed/meta", ocr_needed_dir="ocr_needed"):
    # Ensure directories exist
    os.makedirs(out_text_dir, exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)
    os.makedirs(ocr_needed_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    input_dir = Path(input_dir)
    out_text_dir = Path(out_text_dir)
    out_meta_dir = Path(out_meta_dir)
    ocr_needed_dir = Path(ocr_needed_dir)
    results = []
    for p in tqdm(list(input_dir.glob("**/*.pdf"))):
        res = process_file(p, out_text_dir, out_meta_dir, ocr_needed_dir)
        results.append(res)
    # save summary csv
    df = pd.DataFrame(results)
    df.to_csv("logs/summary.csv", index=False)
    print("Done. Summary saved to logs/summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw")
    parser.add_argument("--out_text", default="processed/texts")
    parser.add_argument("--out_meta", default="processed/meta")
    parser.add_argument("--ocr_needed", default="ocr_needed")
    args = parser.parse_args()
    main(args.input, args.out_text, args.out_meta, args.ocr_needed)
