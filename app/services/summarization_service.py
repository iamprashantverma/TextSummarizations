from sqlalchemy.orm import Session
from app.services.pdf_text_extraction_service import PDFTextExtractionService
from app.models.summary import Summary
from app.providers.base import AIProvider
from app.core.errors import NotFoundError


class SummarizationService:

    def __init__(self, provider: AIProvider):
        self.provider = provider

    async def create_summary(self, db: Session, text: str, max_sentences: int):

        summary_record = Summary(
            text=text,
            max_sentences=max_sentences,
            status="PENDING"
        )

        db.add(summary_record)
        db.commit()
        db.refresh(summary_record)

        try:
            # Run AI summarization
            output = await self.provider.summarize(text, max_sentences)

            summary_record.output = output
            summary_record.status = "COMPLETED"
            db.commit()

            return summary_record

        except Exception as e:
            db.rollback()

            summary_record.status = "FAILED"
            summary_record.error_message = str(e)

            db.add(summary_record)
            db.commit()
            raise

    def get_summaries(self, db: Session):
        return db.query(Summary).order_by(Summary.id.desc()).all()

    def get_summary_by_id(self, db: Session, summary_id: int):
        summary = db.query(Summary).filter(Summary.id == summary_id).first()
        if not summary:
            raise NotFoundError()
        return summary

    async def create_summary_from_pdf(
        self,
        db: Session,
        pdf_bytes: bytes,
        max_sentences: int
    ):
        text = PDFTextExtractionService.extract_text(pdf_bytes)
        return await self.create_summary(db, text, max_sentences)
