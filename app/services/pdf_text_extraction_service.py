import pdfplumber
from io import BytesIO


class PDFTextExtractionService:

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        extracted_text = ""

        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"

        except Exception:
            raise ValueError("Failed to read PDF file")

        cleaned = PDFTextExtractionService._clean_text(extracted_text)

        if not cleaned.strip():
            raise ValueError("PDF contains no readable text")

        return cleaned

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(lines)
