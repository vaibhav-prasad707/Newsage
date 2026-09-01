from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from app.database.connection import Base

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String) # 'ingestion', 'ai_processing', 'market_update'
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    articles_found = Column(Integer, default=0)
    articles_processed = Column(Integer, default=0)
    duplicates = Column(Integer, default=0)
    failures = Column(Integer, default=0)
    status = Column(String) # 'success', 'failed', 'partial_success'
    error_details = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
