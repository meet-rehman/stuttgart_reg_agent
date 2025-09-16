# src/stuttgart_reg_agent/new_crew.py
from pathlib import Path
from crewai import Agent, Crew, Process
from crewai.memory import LongTermMemory
from crewai.memory.storage.rag_storage import RAGStorage

from PyPDF2 import PdfReader

class DocumentCrew:
    """
    Crew that ingests PDF documents into LongTermMemory (RAGStorage)
    for building regulations or other reference data.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.long_term_memory = LongTermMemory(
            storage=RAGStorage(
                embedder_config={"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
                type="long_term",
                path=str(memory_dir / "long_term")
            )
        )

    def _read_pdf(self, pdf_path: Path) -> str:
        """
        Read a PDF and return text content.
        Only works with text-based PDFs, not scanned images.
        """
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            print(f"⚠️ Failed to read {pdf_path}: {e}")
            return ""

    def ingest_pdfs(self, data_dir: Path):
        """
        Find all PDFs in data_dir recursively and add their content to long-term memory.
        """
        pdf_files = list(data_dir.rglob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to ingest.")

        for pdf_path in pdf_files:
            print(f"Ingesting {pdf_path}")
            text = self._read_pdf(pdf_path)
            if text:
                self.long_term_memory.storage.add(
                    doc_id=pdf_path.stem,
                    content=text
                )
        print("✅ PDF ingestion completed.")


# -------------------- Example usage --------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_folder = project_root / "data"  # your folder containing PDFs
    memory_folder = project_root / "memory"

    pdf_crew = DocumentCrew(memory_dir=memory_folder)
    pdf_crew.ingest_pdfs(data_dir=data_folder)
