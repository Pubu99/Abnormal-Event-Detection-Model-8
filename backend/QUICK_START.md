# 🚀 ONE-COMMAND BACKEND SETUP (Professional Way)

## ⚡ **Complete Run Command**

From project root, run:

```powershell
# Activate venv and run backend (can run from anywhere in the project)
cd backend
.\venv\Scripts\Activate
cd api
python app.py
```

**That's it!** The backend now uses absolute paths, so it works from any directory.

---

## 📋 **First Time Setup**

```powershell
# 1. Navigate to project root
cd "E:\ENGINEERING\FOE-UOR\FYP\Model 8\Abnormal-Event-Detection-Model-8"

# 2. Move model file (one time only)
Copy-Item "outputs\checkpoints\best.pth" -Destination "models\best_model.pth" -Force

# 3. Install dependencies in venv (one time only)
cd backend
.\venv\Scripts\Activate
pip install -r requirements.txt
```

---

## 📋 **Quick Reference**

### **Every time you want to run backend:**

```powershell
# From project root
cd backend
.\venv\Scripts\Activate
cd ..
python backend\api\app.py
```

### **To stop backend:**

Press `Ctrl + C`

### **To deactivate venv:**

```powershell
deactivate
```

---

## ✅ **Verification**

Backend is running successfully when you see:

```
✅ Model loaded successfully!
✅ Using device: cuda
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Then check: `http://localhost:8000/health`

---

## 🎯 **Professional Workflow**

**Project Structure:**

```
Project Root (E:\ENGINEERING\FOE-UOR\FYP\Model 8\Abnormal-Event-Detection-Model-8\)
├── backend/
│   ├── venv/           ← Virtual environment HERE
│   └── api/
│       └── app.py      ← Backend code HERE
├── inference/          ← Must be accessible from root
├── src/                ← Must be accessible from root
└── models/             ← Must be accessible from root
```

**Why run from root?**

- `inference/` imports `src.models.research_model`
- `src/` imports are relative to project root
- `models/` path is relative to project root
- Venv is in `backend/` but code runs from root

**Command breakdown:**

1. `cd backend` - Go to backend folder
2. `.\venv\Scripts\Activate` - Activate virtual environment
3. `cd ..` - **Go back to project root** (CRITICAL!)
4. `python backend\api\app.py` - Run from root, accessing venv's Python

---

## 🔥 **All Dependencies Installed**

✅ PyTorch 2.7.0+cu128
✅ FastAPI + Uvicorn
✅ OpenCV
✅ YOLO (Ultralytics)
✅ Albumentations
✅ TensorBoard
✅ Loguru
✅ EfficientNet
✅ TorchMetrics
✅ Hydra
✅ Rich
✅ WandB
✅ Plotly
✅ Imbalanced-learn
✅ ImageIO
✅ PyNVML
✅ And all other dependencies!

---

## 🎉 **You're Professional Now!**

- ✅ Virtual environment in `backend/venv/`
- ✅ All dependencies in `requirements.txt`
- ✅ Running from correct directory (project root)
- ✅ Imports work correctly
- ✅ Model file in correct location

This is how **production systems** are structured!
