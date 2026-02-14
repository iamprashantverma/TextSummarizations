from pydantic import BaseModel
from typing import Optional

class SummarizeRequest(BaseModel):
    text: str
    max_sentences: int


class SummarizeResponse(BaseModel):
    id: int
    text: str
    max_sentences: int
    output: Optional[str]
    status: str

    class Config:
        from_attributes = True
