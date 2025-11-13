"""
Simple Reinforcement Learning service for integrating user feedback.

This file implements a lightweight contextual-bandit / policy model
using PyTorch. It stores experiences to a JSONL file and can be trained
periodically or on demand. The policy is a small MLP that predicts the
probability a detection is a true anomaly given the fused features.

Design decisions (kept simple for safety):
- Use supervised-style training from (features, reward) where reward
  is +1 for confirmed anomaly, -1 for decline / not_anomaly.
- Treat it as binary classification (reward > 0 => positive). Use BCE
  loss to optimize the policy output (probability of true anomaly).
- Store experiences in `backend/data/rl_experiences.jsonl` as lines of
  JSON. Each experience contains features, action, reward, timestamp.

Integration:
- Call `log_experience()` from the fusion engine when user feedback is
  received (apply_feedback / decline_all).
- Use `select_action(features)` to ask the agent whether to accept a
  detection (useful for dynamic thresholds).

This is intentionally a scaffold: it provides a working loop and can be
extended to more advanced RL methods later (policy-gradients, PPO,
or actor-critic) once more feedback and compute are available.
"""

import os
import json
import sqlite3
import time
from typing import Dict, Any, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
EXPERIENCE_FILE = os.path.join(DATA_DIR, "rl_experiences.jsonl")
DB_FILE = os.path.join(DATA_DIR, "rl_experiences.db")
MODEL_FILE = os.path.join(DATA_DIR, "rl_policy.pt")


def _default_device():
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    def __init__(self, input_dim: int = 8, hidden: int = 64):
        """
        Policy network for anomaly detection.
        
        Args:
            input_dim: Input feature dimension (8 for basic, 16 for contextual)
            hidden: Hidden layer size (increased to 64 for better capacity)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    """Lightweight RL agent acting as a contextual bandit / classifier."""

    def __init__(self, input_dim: int = 16, lr: float = 1e-3, device=None, db_file: str = None, model_file: str = None):
        """
        Initialize RL agent.
        
        Args:
            input_dim: Feature dimension (16 for contextual, 8 for legacy)
            lr: Learning rate
            device: torch device
            db_file: Database file path
            model_file: Model checkpoint path
        """
        self.device = device or _default_device()
        self.input_dim = input_dim
        self.model = None
        self.optimizer = None
        self.lr = lr
        # allow override of DB and model file when training on snapshots
        self.db_file = db_file or DB_FILE
        self.model_file = model_file or MODEL_FILE
        self._ensure_model()

    def _ensure_model(self):
        if torch is None:
            print("[rl_agent] PyTorch not available; RLAgent disabled")
            return
        if self.model is None:
            self.model = SimplePolicy(input_dim=self.input_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Ensure DB initialized
        try:
            self._init_db()
        except Exception:
            pass

    def _read_experiences(self) -> List[Dict[str, Any]]:
        # Read experiences from SQLite DB; if DB not present, try to read JSONL
        if os.path.exists(self.db_file):
            exps = []
            try:
                conn = sqlite3.connect(self.db_file)
                cur = conn.cursor()
                cur.execute("SELECT id, ts, features, action, reward, meta FROM experiences ORDER BY id DESC")
                rows = cur.fetchall()
                for r in rows:
                    _id, ts, features_j, action, reward, meta_j = r
                    try:
                        features = json.loads(features_j) if features_j else {}
                    except Exception:
                        features = {}
                    try:
                        meta = json.loads(meta_j) if meta_j else {}
                    except Exception:
                        meta = {}
                    exps.append({'id': _id, 'ts': ts, 'features': features, 'action': action, 'reward': reward, 'meta': meta})
                conn.close()
            except Exception:
                return []
            return exps

        # Fallback to legacy JSONL
        if not os.path.exists(EXPERIENCE_FILE):
            return []
        exps = []
        with open(EXPERIENCE_FILE, 'r') as f:
            for line in f:
                try:
                    exps.append(json.loads(line))
                except Exception:
                    continue
        return exps

    def log_experience(self, features: Dict[str, float], action: int, reward: float, meta: Dict = None):
        """Append an experience to the JSONL store.

        features: mapping of feature name to float
        action: int (0/1) - action taken (e.g., 1=reported/flagged, 0=not)
        reward: float reward signal from user feedback
        meta: optional dict with extra info (detection_id, timestamp)
        """
        # Insert into SQLite DB (preferred)
        try:
            self._init_db()
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            ts = float(time.time())
            cur.execute(
                "INSERT INTO experiences (ts, features, action, reward, meta) VALUES (?, ?, ?, ?, ?)",
                (ts, json.dumps(features), int(action), float(reward), json.dumps(meta or {}))
            )
            conn.commit()
            conn.close()
            return
        except Exception:
            # Fallback to JSONL if DB fails
            exp = {
                'ts': time.time(),
                'features': features,
                'action': int(action),
                'reward': float(reward),
                'meta': meta or {}
            }
            with open(EXPERIENCE_FILE, 'a') as f:
                f.write(json.dumps(exp) + "\n")

    def _build_dataset(self):
        exps = self._read_experiences()
        X = []
        y = []
        for e in exps:
            feats = e.get('features', {})
            vec = self._feats_to_vector(feats)
            X.append(vec)
            # convert reward to binary label: reward > 0 => positive
            label = 1.0 if e.get('reward', 0.0) > 0 else 0.0
            y.append(label)
        if not X:
            return None, None
        import numpy as _np
        return _np.array(X, dtype=_np.float32), _np.array(y, dtype=_np.float32)

    def _init_db(self):
        """Create SQLite DB and experiences table if not present. Also migrate JSONL if found."""
        try:
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                features TEXT,
                action INTEGER,
                reward REAL,
                meta TEXT
            )
            """)
            conn.commit()
            conn.close()
        except Exception:
            return

        # Migrate legacy JSONL into DB (one-time)
        if os.path.exists(EXPERIENCE_FILE):
            try:
                with open(EXPERIENCE_FILE, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    return
                conn = sqlite3.connect(self.db_file)
                cur = conn.cursor()
                for line in lines:
                    try:
                        e = json.loads(line)
                        cur.execute(
                            "INSERT INTO experiences (ts, features, action, reward, meta) VALUES (?, ?, ?, ?, ?)",
                            (float(e.get('ts', time.time())), json.dumps(e.get('features', {})), int(e.get('action', 1)), float(e.get('reward', 0.0)), json.dumps(e.get('meta', {})))
                        )
                    except Exception:
                        continue
                conn.commit()
                conn.close()
                # Optionally keep JSONL for audit; do not delete automatically
            except Exception:
                pass

    def get_experiences(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent experiences from DB as list of dicts."""
        exps = []
        try:
            self._init_db()
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute("SELECT id, ts, features, action, reward, meta FROM experiences ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            for r in rows:
                _id, ts, features_j, action, reward, meta_j = r
                try:
                    features = json.loads(features_j) if features_j else {}
                except Exception:
                    features = {}
                try:
                    meta = json.loads(meta_j) if meta_j else {}
                except Exception:
                    meta = {}
                exps.append({'id': _id, 'ts': ts, 'features': features, 'action': action, 'reward': reward, 'meta': meta})
            conn.close()
        except Exception:
            return []
        return exps

    def _feats_to_vector(self, features: Dict[str, float]):
        """
        Convert feature dict to vector.
        
        Supports two modes:
        1. Basic (8 features): Original fusion scores
        2. Contextual (16 features): Includes tracking data
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            List of float values matching input_dim
        """
        # Basic features (always present)
        basic_order = [
            'fusion_score', 'ml_score', 'object_score', 'pose_score', 'motion_score',
            'consensus_count', 'confidence', 'severity'
        ]
        
        # Contextual features (from advanced tracking)
        contextual_order = [
            'track_duration',        # How long object has been tracked (seconds)
            'movement_speed',        # Movement speed (pixels/frame)
            'loitering_score',       # Loitering likelihood (0-1)
            'track_confidence',      # Track quality (0-1)
            'gesture_score',         # Gesture threat score (0-1)
            'held_object_count',     # Number of held objects
            'body_pose_score',       # Body pose anomaly score (0-1)
            'temporal_consistency'   # Consistency over time (0-1)
        ]
        
        # Build vector based on input_dim
        if self.input_dim == 16:
            # Contextual mode: use all 16 features
            order = basic_order + contextual_order
        else:
            # Legacy mode: use basic 8 features only
            order = basic_order
            
        vec = [float(features.get(k, 0.0) or 0.0) for k in order]
        
        # Pad or truncate to match input_dim
        if len(vec) < self.input_dim:
            vec.extend([0.0] * (self.input_dim - len(vec)))
        elif len(vec) > self.input_dim:
            vec = vec[:self.input_dim]
            
        return vec

    def train(self, epochs: int = 10, batch_size: int = 32, model_path: str = None):
        """Train the policy on logged experiences.

        Returns number of samples used.
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")

        X, y = self._build_dataset()
        if X is None or len(X) == 0:
            return 0

        import numpy as _np
        self._ensure_model()
        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.BCELoss()
        self.model.train()
        for ep in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
        # save model (atomic behavior handled by caller if desired)
        if model_path:
            # save to provided model_path
            self.save(model_path)
        else:
            self.save()
        return len(X)

    def get_experience_count(self) -> int:
        """Return total number of stored experiences."""
        try:
            self._init_db()
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM experiences")
            c = cur.fetchone()[0]
            conn.close()
            return int(c or 0)
        except Exception:
            # fallback to JSONL count
            try:
                if os.path.exists(EXPERIENCE_FILE):
                    with open(EXPERIENCE_FILE, 'r') as f:
                        return sum(1 for _ in f)
            except Exception:
                return 0
            return 0

    def get_max_experience_id(self) -> int:
        """Return the maximum experience id in DB (0 if none)."""
        try:
            self._init_db()
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute("SELECT MAX(id) FROM experiences")
            r = cur.fetchone()[0]
            conn.close()
            return int(r or 0)
        except Exception:
            return 0

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Return probability (0..1) that detection is a true anomaly."""
        if torch is None or self.model is None:
            return 0.5
        vec = self._feats_to_vector(features)
        import numpy as _np
        xb = torch.from_numpy(_np.array([vec], dtype=_np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            p = self.model(xb).cpu().numpy()[0, 0]
        return float(p)

    def select_action(self, features: Dict[str, float], threshold: float = 0.5) -> int:
        """Simple policy: return 1 if prob >= threshold else 0."""
        p = self.predict_proba(features)
        return 1 if p >= threshold else 0

    def save(self, path: str = None):
        """Save model state to given path atomically (overwrite).

        If path is None, the canonical MODEL_FILE is used.
        """
        if torch is None or self.model is None:
            return
        if path is None:
            path = self.model_file
        # write to temp file and atomically replace
        tmp = path + ".tmp" + str(int(time.time()))
        try:
            torch.save(self.model.state_dict(), tmp)
            try:
                os.replace(tmp, path)
            except Exception:
                # fallback to rename
                os.rename(tmp, path)
        except Exception:
            # best-effort save; ignore errors here
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def load(self):
        if torch is None:
            return
        if not os.path.exists(self.model_file):
            return
        self._ensure_model()
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))


# Singleton agent instance for easy import/use across services
_AGENT = None


def get_agent(use_contextual: bool = True) -> RLAgent:
    """
    Get singleton RL agent instance.
    
    Args:
        use_contextual: If True, use 16-feature contextual mode. If False, use 8-feature legacy mode.
        
    Returns:
        RLAgent instance
    """
    global _AGENT
    if _AGENT is None:
        # Use contextual mode (16 features) by default
        input_dim = 16 if use_contextual else 8
        _AGENT = RLAgent(input_dim=input_dim, db_file=DB_FILE, model_file=MODEL_FILE)
        try:
            _AGENT.load()
        except Exception:
            pass
    return _AGENT


def _train_process_target(epochs: int, batch_size: int, niceness: int, tmp_model_path: str):
    """Entrypoint for child training process.

    Sets niceness (best-effort), creates a fresh RLAgent instance and trains.
    Saves the trained model to tmp_model_path on success.
    The parent process should verify tmp_model_path and then atomically move it to the canonical model path.
    """
    try:
        try:
            os.nice(int(niceness))
        except Exception:
            # may require privileges; ignore if fails
            pass

        # Helper: compute metrics from true labels and predicted probabilities
        def _compute_metrics(y_true_np, y_prob_np, threshold=0.5):
            """Return a dict with accuracy, precision, recall, f1, auc (if available) and val_loss placeholder."""
            metrics = {}
            try:
                # try to use sklearn if available
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                y_pred_labels = (y_prob_np >= threshold).astype(int)
                metrics['accuracy'] = float(accuracy_score(y_true_np, y_pred_labels))
                # precision/recall may be ill-defined if no positives; use zero_division=0
                metrics['precision'] = float(precision_score(y_true_np, y_pred_labels, zero_division=0))
                metrics['recall'] = float(recall_score(y_true_np, y_pred_labels, zero_division=0))
                metrics['f1'] = float(f1_score(y_true_np, y_pred_labels, zero_division=0))
                try:
                    metrics['auc'] = float(roc_auc_score(y_true_np, y_prob_np))
                except Exception:
                    metrics['auc'] = None
            except Exception:
                # fallback: implement simple metrics
                try:
                    y_pred_labels = (y_prob_np >= threshold).astype(int)
                    tp = int(((y_pred_labels == 1) & (y_true_np == 1)).sum())
                    tn = int(((y_pred_labels == 0) & (y_true_np == 0)).sum())
                    fp = int(((y_pred_labels == 1) & (y_true_np == 0)).sum())
                    fn = int(((y_pred_labels == 0) & (y_true_np == 1)).sum())
                    total = tp + tn + fp + fn
                    metrics['accuracy'] = float((tp + tn) / total) if total > 0 else 0.0
                    metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    prec = metrics['precision']
                    rec = metrics['recall']
                    metrics['f1'] = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
                    metrics['auc'] = None
                except Exception:
                    metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': None}
            return metrics

        # Create a snapshot of the canonical DB to avoid races while writing
        try:
            import shutil
            ts = int(time.time())
            snapshot_db = DB_FILE + f".snapshot.{ts}.db"
            if os.path.exists(DB_FILE):
                shutil.copy(DB_FILE, snapshot_db)
            else:
                snapshot_db = DB_FILE  # fallback if DB not present
        except Exception:
            snapshot_db = DB_FILE

        # Use the snapshot DB for training and validation
        agent = RLAgent(input_dim=8, db_file=snapshot_db, model_file=tmp_model_path)

        # Build dataset from snapshot
        try:
            X, y = agent._build_dataset()
            if X is None or len(X) == 0:
                return

            import numpy as _np
            # Shuffle and split train/validation
            n = len(X)
            idx = _np.random.permutation(n)
            val_frac = 0.1
            val_size = max(1, int(n * val_frac))
            val_idx = idx[:val_size]
            train_idx = idx[val_size:]

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            # Convert to tensors
            import torch as _torch
            X_t = _torch.from_numpy(X_train).to(agent.device)
            y_t = _torch.from_numpy(y_train).unsqueeze(1).to(agent.device)
            dataset = _torch.utils.data.TensorDataset(X_t, y_t)
            loader = _torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            loss_fn = nn.BCELoss()
            agent._ensure_model()
            agent.model.train()
            optimizer = optim.Adam(agent.model.parameters(), lr=agent.lr)
            for ep in range(epochs):
                for xb, yb in loader:
                    pred = agent.model(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            agent.model.eval()
            with _torch.no_grad():
                Xv = _torch.from_numpy(X_val).to(agent.device)
                yv = _torch.from_numpy(y_val).unsqueeze(1).to(agent.device)
                preds = agent.model(Xv)
                # compute val loss and accuracy
                val_loss = float(loss_fn(preds, yv).cpu().numpy())
                pred_labels = (preds.cpu().numpy() >= 0.5).astype(int)
                yv_np = y_val.reshape(-1, 1).astype(int)
                val_acc = float((pred_labels == yv_np).mean()) if len(yv_np) > 0 else 0.0

            # Save trained model to tmp path
            try:
                agent.save(tmp_model_path)
            except Exception:
                pass

            # Write metrics file next to tmp model
            try:
                metrics = {
                    'ts': int(time.time()),
                    'val_samples': int(len(y_val)),
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_acc),
                    'trained_samples': int(len(y_train))
                }
                with open(tmp_model_path + '.metrics.json', 'w') as mf:
                    json.dump(metrics, mf)
            except Exception:
                pass

        finally:
            # cleanup snapshot DB to avoid accumulating files
            try:
                if snapshot_db != DB_FILE and os.path.exists(snapshot_db):
                    os.remove(snapshot_db)
            except Exception:
                pass
    except Exception as e:
        # Ensure the process exits cleanly on error; parent will detect missing tmp file
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
    finally:
        return

