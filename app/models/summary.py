from sqlalchemy import (
    Column,
    Integer,
    Text,
    DateTime,
    Enum,
    func,
)
from app.db.base import Base


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    max_sentences = Column(Integer, nullable=False)
    output = Column(Text, nullable=True)
    status = Column(
        Enum("PENDING", "COMPLETED", "FAILED", name="summary_status"),
        default="PENDING",
        nullable=False,
    )
    created_at = Column(DateTime, server_default=func.now())
