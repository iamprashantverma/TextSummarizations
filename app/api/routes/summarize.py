from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session

from app.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.services.summarization_service import SummarizationService
from app.providers.huggingface_provider import HuggingFaceProvider
from app.api.deps import get_db

router = APIRouter()

provider = HuggingFaceProvider()
service = SummarizationService(provider)


@router.post("/summarize", response_model=SummarizeResponse)
async def create_summary(
    request: SummarizeRequest,
    db: Session = Depends(get_db)
):
    return await service.create_summary(
        db=db,
        text=request.text,
        max_sentences=request.max_sentences
    )


@router.post("/summarize/pdf", response_model=SummarizeResponse)
async def create_summary_from_pdf(
    file: UploadFile = File(...),
    max_sentences: int = Form(5),
    db: Session = Depends(get_db)
):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")

    pdf_bytes = await file.read()
    return await service.create_summary_from_pdf( db=db, pdf_bytes=pdf_bytes, max_sentences=max_sentences)


@router.get("/summarize")
def get_summaries(db: Session = Depends(get_db)):
    return service.get_summaries(db)


@router.get("/summarize/{summary_id}")
def get_summary(summary_id: int, db: Session = Depends(get_db)):
    return service.get_summary_by_id(db, summary_id)
