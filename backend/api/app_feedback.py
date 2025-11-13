# app_feedback.py
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "postgresql://postgres:password@db:5432/fyp_db"  # change for local
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True)
    camera_id = Column(String, nullable=False)
    frame_timestamp = Column(DateTime, nullable=False)
    frame_path = Column(String)
    detection_label = Column(String)
    detection_confidence = Column(Float)
    bbox = Column(JSON)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey("detections.id", ondelete="CASCADE"))
    user_id = Column(String)
    feedback_label = Column(String)
    feedback_type = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    detection = relationship("Detection")

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Pydantic schemas
class DetectionIn(BaseModel):
    camera_id: str
    frame_timestamp: datetime
    frame_path: Optional[str] = None
    detection_label: Optional[str] = None
    detection_confidence: Optional[float] = None
    bbox: Optional[Any] = None
    model_version: str

class FeedbackIn(BaseModel):
    user_id: Optional[str] = None
    feedback_label: str
    feedback_type: str  # "confirm", "correct", "ignore"
    notes: Optional[str] = None

@app.post("/api/detections")
def create_detection(d: DetectionIn):
    db = SessionLocal()
    det = Detection(
        camera_id=d.camera_id,
        frame_timestamp=d.frame_timestamp,
        frame_path=d.frame_path,
        detection_label=d.detection_label,
        detection_confidence=d.detection_confidence,
        bbox=d.bbox,
        model_version=d.model_version
    )
    db.add(det)
    db.commit()
    db.refresh(det)
    db.close()
    return {"id": det.id, "message": "detection stored"}

@app.post("/api/detections/{detection_id}/feedback")
def submit_feedback(detection_id: int = Path(...), fb: FeedbackIn = None):
    db = SessionLocal()
    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det:
        db.close()
        raise HTTPException(status_code=404, detail="Detection not found")
    feedback = Feedback(
        detection_id=detection_id,
        user_id=fb.user_id,
        feedback_label=fb.feedback_label,
        feedback_type=fb.feedback_type,
        notes=fb.notes
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    db.close()
    return {"id": feedback.id, "message": "feedback stored"}

@app.get("/api/feedbacks")
def get_feedbacks(limit: int = 100, offset: int = 0):
    db = SessionLocal()
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).limit(limit).offset(offset).all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "detection_id": r.detection_id,
            "feedback_label": r.feedback_label,
            "feedback_type": r.feedback_type,
            "notes": r.notes,
            "created_at": r.created_at,
            "detection": {
                "camera_id": r.detection.camera_id,
                "frame_path": r.detection.frame_path,
                "detection_label": r.detection.detection_label,
                "detection_confidence": r.detection.detection_confidence,
                "bbox": r.detection.bbox
            }
        })
    db.close()
    return out
