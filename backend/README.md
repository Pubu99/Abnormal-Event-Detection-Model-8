# 🚀 Backend Setup & Run Guide

Complete guide to set up and run the Anomaly Detection backend API.

---

## 📋 **Prerequisites**

- Python 3.9+ installed
- CUDA 12.8 (for GPU acceleration with RTX 5090)
- Trained model file: `models/best_model.pth`

---

## ⚡ **Quick Start (3 Steps)**

### **1. Install Dependencies**

```powershell
# IMPORTANT: Run from PROJECT ROOT, not from backend folder!
# The backend needs to import from inference/, src/, etc.

cd backend

# Install PyTorch with CUDA first (IMPORTANT!)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install all other dependencies
pip install -r requirements.txt
```

### **2. Move Model File (If Not Done)**

```powershell
# From project root
Copy-Item "outputs\checkpoints\best.pth" -Destination "models\best_model.pth"
```

### **3. Run Backend Server**

```powershell
# ⚠️ IMPORTANT: Run from PROJECT ROOT!
# Navigate back to project root if you're in backend folder
cd ..

# Run backend from project root
python backend\api\app.py
```

✅ **Backend will be running at:** `http://localhost:8000`

---

## ⚠️ **CRITICAL: Path Structure**

The backend must be run from the **PROJECT ROOT** because:

- `inference/` folder is at root level
- `src/` folder is at root level
- `models/` folder is at root level
- `backend/` folder contains the API code

**Correct structure:**

```
Project Root/
├── backend/          # API code
│   ├── venv/         # Virtual environment
│   └── api/
│       └── app.py    # FastAPI app
├── inference/        # Inference engine
├── src/              # Source code (models, utils)
└── models/           # Trained models
```

**Run from:** Project Root
**Command:** `python backend\api\app.py`

---

## 🔧 **Professional Setup (With Virtual Environment)**

### **Step 1: Create Virtual Environment**

```powershell
cd backend

# Create venv
python -m venv venv

# Activate venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### **Step 2: Install PyTorch with CUDA**

```powershell
# This MUST be installed FIRST (before requirements.txt)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

⏱️ **Time:** ~5-10 minutes (~2GB download)

### **Step 3: Install All Dependencies**

```powershell
pip install -r requirements.txt
```

### **Step 4: Verify Installation**

```powershell
# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Expected Output:**

```
PyTorch: 2.7.0+cu128
CUDA: True
Device: NVIDIA GeForce RTX 5090
```

### **Step 5: Run Backend**

```powershell
cd api
python app.py
```

---

## 🌐 **Backend Endpoints**

Once running, the backend provides:

| Endpoint                            | Method    | Description                   |
| ----------------------------------- | --------- | ----------------------------- |
| `http://localhost:8000/`            | GET       | Welcome message               |
| `http://localhost:8000/health`      | GET       | Health check & model status   |
| `http://localhost:8000/api/predict` | POST      | Upload video for analysis     |
| `http://localhost:8000/ws/stream`   | WebSocket | Real-time streaming           |
| `http://localhost:8000/docs`        | GET       | Interactive API documentation |

---

## 📚 **Complete Installation Commands**

### **Option A: Global Environment (Fast)**

```powershell
cd backend

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Run backend
cd api
python app.py
```

### **Option B: Virtual Environment (Recommended)**

```powershell
cd backend

# Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Run backend
cd api
python app.py
```

---

## 🧪 **Testing the Backend**

### **1. Check Health Endpoint**

```powershell
# In browser or using curl
curl http://localhost:8000/health
```

**Expected Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### **2. Test Video Upload**

```powershell
# Using PowerShell
$uri = "http://localhost:8000/api/predict"
$filePath = "path\to\test\video.mp4"
$form = @{
    file = Get-Item -Path $filePath
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

### **3. View API Documentation**

Open in browser: `http://localhost:8000/docs`

---

## 🐛 **Troubleshooting**

### **Problem: PyTorch DLL Error**

```powershell
# Uninstall PyTorch
pip uninstall torch torchvision torchaudio -y

# Reinstall with CUDA
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### **Problem: Module Not Found**

```powershell
# Reinstall requirements
pip install -r requirements.txt
```

### **Problem: CUDA Not Available**

```powershell
# Check PyTorch version
pip show torch

# Should show: torch (2.7.0+cu128)
# If not, reinstall PyTorch with correct CUDA version
```

### **Problem: Model File Not Found**

```powershell
# Copy model from outputs to models folder
Copy-Item "outputs\checkpoints\best.pth" -Destination "models\best_model.pth"
```

### **Problem: Port 8000 Already in Use**

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or change port in app.py (line with uvicorn.run)
```

---

## 📦 **What Gets Installed**

| Package        | Version     | Purpose                 |
| -------------- | ----------- | ----------------------- |
| torch          | 2.7.0+cu128 | Deep learning framework |
| torchvision    | 0.22.0      | Vision utilities        |
| fastapi        | 0.109.2     | Web framework           |
| uvicorn        | 0.27.1      | ASGI server             |
| opencv-python  | 4.10.0.84   | Video processing        |
| ultralytics    | 8.3.50      | YOLO object detection   |
| albumentations | 1.4.20      | Image augmentation      |
| numpy          | 1.26.4      | Numerical computing     |
| pandas         | 2.2.3       | Data processing         |

**Total Size:** ~3GB

---

## 🎯 **Development Workflow**

### **Every time you start working:**

```powershell
# Navigate to backend
cd backend

# Activate virtual environment (if using venv)
.\venv\Scripts\Activate

# Start backend
cd api
python app.py
```

### **To stop backend:**

Press `Ctrl + C` in the terminal

### **To deactivate virtual environment:**

```powershell
deactivate
```

---

## 🚀 **Production Deployment**

### **Run with Gunicorn (Linux/Mac):**

```bash
cd backend/api
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **Run as Background Service (Windows):**

```powershell
# Using nssm (Non-Sucking Service Manager)
nssm install AnomalyDetectionAPI "C:\path\to\python.exe" "C:\path\to\backend\api\app.py"
nssm start AnomalyDetectionAPI
```

---

## 📊 **Performance Tips**

- **GPU Acceleration:** Ensure CUDA is available (`torch.cuda.is_available()` returns `True`)
- **Batch Processing:** Backend processes videos in batches of 16 frames
- **Model Loading:** Model loads once at startup (~5 seconds)
- **Video Processing:** ~1-2 seconds per video (depending on length)

---

## 🔒 **Security Notes**

- **File Upload Limits:** Max 500MB per video
- **Allowed Formats:** MP4, AVI, MOV, MKV
- **CORS:** Enabled for `http://localhost:3000` (frontend)
- **Production:** Update CORS origins in `app.py` for production deployment

---

## 📝 **Project Structure**

```
backend/
├── api/
│   ├── app.py              # Main FastAPI application
│   └── routes/             # API route handlers
├── core/                   # Core business logic
├── services/               # External services integration
├── requirements.txt        # Python dependencies
├── INSTALL.md             # Detailed installation guide
└── README.md              # This file
```

---

## ✅ **Checklist Before Running**

- [ ] Python 3.9+ installed
- [ ] PyTorch 2.7.0+cu128 installed
- [ ] All requirements from `requirements.txt` installed
- [ ] Model file exists at `models/best_model.pth`
- [ ] Port 8000 is available
- [ ] CUDA is available (for GPU acceleration)

---

## 🎉 **Quick Reference**

```powershell
# Install everything
cd backend
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Run backend
cd api
python app.py

# Open in browser
start http://localhost:8000/docs
```

---

## 📞 **Need Help?**

- **API Documentation:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Check logs:** Console output shows all requests and errors

---

**🎯 Backend is ready when you see:**

```
✅ Model loaded successfully!
✅ Using device: cuda
✅ Model device: cuda
INFO:     Uvicorn running on http://0.0.0.0:8000
```
