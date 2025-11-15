"""
Simple persistent store for confirmed anomalies using SQLite.

Provides:
- init_db(path)
- save_confirmed_detection(record)
- get_confirmed_detections(limit=100, date=None)

Records are stored with JSON for complex fields so the schema is flexible.
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_FILE = Path(__file__).parent.parent / 'data' / 'confirmed_anomalies.db'


def _ensure_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id TEXT,
            recorded_at_utc TEXT,
            anomaly_type TEXT,
            severity TEXT,
            fusion_score REAL,
            confidence REAL,
            ml_score REAL,
            object_score REAL,
            pose_score REAL,
            motion_score REAL,
            detected_objects TEXT,
            bounding_boxes TEXT,
            metadata TEXT,
            user TEXT,
            comment TEXT
        )
        '''
    )
    conn.commit()


def _get_conn(path: Optional[Path] = None) -> sqlite3.Connection:
    p = path or DB_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    _ensure_db(conn)
    return conn


def save_confirmed_detection(record: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Persist a confirmed detection record. Returns the inserted row id."""
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO anomalies (
            detection_id, recorded_at_utc, anomaly_type, severity,
            fusion_score, confidence, ml_score, object_score, pose_score, motion_score,
            detected_objects, bounding_boxes, metadata, user, comment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            record.get('detection_id'),
            record.get('timestamp'),
            record.get('anomaly_type'),
            record.get('severity'),
            float(record.get('fusion_score') or 0.0),
            float(record.get('confidence') or 0.0),
            float(record.get('ml_score') or 0.0),
            float(record.get('object_score') or 0.0),
            float(record.get('pose_score') or 0.0),
            float(record.get('motion_score') or 0.0),
            json.dumps(record.get('detected_objects') or []),
            json.dumps(record.get('bounding_boxes') or []),
            json.dumps(record.get('metadata') or {}),
            record.get('user'),
            record.get('comment')
        )
    )
    conn.commit()
    rowid = cur.lastrowid
    try:
        conn.close()
    except Exception:
        pass
    return rowid


def get_confirmed_detections(limit: int = 100, date: Optional[str] = None, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return confirmed anomalies. If date is provided (YYYY-MM-DD) filter by that date (UTC)."""
    conn = _get_conn(db_path)
    cur = conn.cursor()
    if date:
        like = f"{date}%"
        cur.execute('SELECT * FROM anomalies WHERE recorded_at_utc LIKE ? ORDER BY id DESC LIMIT ?', (like, limit))
    else:
        cur.execute('SELECT * FROM anomalies ORDER BY id DESC LIMIT ?', (limit,))
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    results: List[Dict[str, Any]] = []
    for r in rows:
        rec = dict(zip(cols, r))
        # parse JSON fields
        try:
            rec['detected_objects'] = json.loads(rec.get('detected_objects') or '[]')
        except Exception:
            rec['detected_objects'] = []
        try:
            rec['bounding_boxes'] = json.loads(rec.get('bounding_boxes') or '[]')
        except Exception:
            rec['bounding_boxes'] = []
        try:
            rec['metadata'] = json.loads(rec.get('metadata') or '{}')
        except Exception:
            rec['metadata'] = {}
        results.append(rec)
    try:
        conn.close()
    except Exception:
        pass
    return results
