from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database.connection import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    model = Column(String, nullable=False)
    prediction_date = Column(DateTime(timezone=True), server_default=func.now())
    target_date = Column(DateTime(timezone=True))
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)
    confidence = Column(Float)
    feature_version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    model = Column(String, nullable=False)
    training_period_start = Column(DateTime(timezone=True))
    training_period_end = Column(DateTime(timezone=True))
    validation_period_start = Column(DateTime(timezone=True))
    validation_period_end = Column(DateTime(timezone=True))
    metrics = Column(String) # JSON string of metrics
    parameters = Column(String) # JSON string of params
    created_at = Column(DateTime(timezone=True), server_default=func.now())
