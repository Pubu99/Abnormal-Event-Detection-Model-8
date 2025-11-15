"""Firebase helper for uploading evidence images and pushing events to Firestore.

Usage:
  - Set environment variables:
      FIREBASE_SERVICE_ACCOUNT=path/to/serviceAccount.json
      FIREBASE_BUCKET=your-project-id.appspot.com
  - Call init_firebase() once, then use upload_image_bytes() and push_event().

This file does not attempt to run if credentials are missing; it raises clear errors
so you can provide the service account file before calling from the ML pipeline.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
except Exception:  # pragma: no cover - runtime dependency
    firebase_admin = None


def init_firebase(service_account_path: Optional[str] = None, bucket_name: Optional[str] = None):
    """Initialize Firebase Admin SDK and return (firestore_client, storage_bucket).

    The function reads `FIREBASE_SERVICE_ACCOUNT` and `FIREBASE_BUCKET` env vars
    if explicit values are not provided.
    """
    if firebase_admin is None:
        raise RuntimeError("firebase_admin package is not installed. Please install firebase-admin in your environment.")

    sa_path = service_account_path or os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not sa_path or not os.path.exists(sa_path):
        raise RuntimeError("Firebase service account JSON not found. Set FIREBASE_SERVICE_ACCOUNT to the path of the JSON file.")

    bucket = bucket_name or os.getenv("FIREBASE_BUCKET")
    if not bucket:
        raise RuntimeError("FIREBASE_BUCKET environment variable is required (your-project-id.appspot.com)")

    # Initialize app only once
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket})

    db = firestore.client()
    bkt = storage.bucket()
    return db, bkt


def upload_image_bytes(bucket, image_bytes: bytes, dest_path: Optional[str] = None, content_type: str = "image/jpeg") -> str:
    """Upload raw image bytes to Firebase Storage and make it public.

    Args:
        bucket: firebase_admin.storage.bucket.Bucket instance
        image_bytes: Raw bytes of the image (JPEG/PNG)
        dest_path: Path in the bucket, e.g. 'uploads/events/{event_id}/snapshot.jpg'

    Returns:
        Public URL of the uploaded object.
    """
    if dest_path is None:
        dest_path = f"uploads/events/{uuid.uuid4().hex}/snapshot.jpg"

    blob = bucket.blob(dest_path)
    blob.upload_from_string(image_bytes, content_type=content_type)
    # Make public (note: you can use signed URLs instead for production)
    try:
        blob.make_public()
        return blob.public_url
    except Exception:
        # If making public fails (restricted buckets), return gs:// path as fallback
        return f"gs://{bucket.name}/{dest_path}"


def push_event(db, payload: Dict[str, Any], collection: str = "events") -> str:
    """Push an event document into Firestore and return the new document ID.

    The payload should be a JSON-serializable dict.
    """
    if db is None:
        raise RuntimeError("Firestore client not initialized")

    doc_ref = db.collection(collection).document()
    # Ensure a timestamp is present
    payload = dict(payload)
    if "timestamp" not in payload:
        payload["timestamp"] = firestore.SERVER_TIMESTAMP

    doc_ref.set(payload)
    return doc_ref.id


def build_event_payload(camera_id: str, anomaly_type: str, fusion_score: float, image_url: Optional[str] = None, extra: Optional[Dict] = None) -> Dict[str, Any]:
    payload = {
        "cameraId": camera_id,
        "type": anomaly_type,
        "fusion_score": float(fusion_score),
        "severity": determine_severity(fusion_score),
        "source": "ml",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if image_url:
        payload["imageUrl"] = image_url
    if extra:
        payload.setdefault("metadata", {}).update(extra)
    return payload


def determine_severity(score: float) -> str:
    if score >= 0.7:
        return "CRITICAL"
    if score >= 0.5:
        return "ABNORMAL"
    if score >= 0.3:
        return "SUSPICIOUS"
    return "NORMAL"
