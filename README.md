# 🎥 Abnormal Event Detection System - Professional Multi-Modal Intelligence

**Production-Ready Real-Time Anomaly Detection with Intelligent Fusion Engine**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React 19.2](https://img.shields.io/badge/React-19.2-blue.svg)](https://react.dev/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.38%25-brightgreen.svg)](docs/RESULTS_AND_ANALYSIS.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Enterprise-grade full-stack anomaly detection system combining a 99.38% accurate deep learning model (EfficientNet-B0 + BiLSTM + Transformer) with 6 real-time detection modalities, intelligent multi-modal fusion, FastAPI WebSocket backend, and React-based professional frontend for comprehensive video surveillance.**

---

## 📋 Table of Contents

- [🎯 System Overview](#-system-overview)
- [🌟 Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📡 API Documentation](#-api-documentation)
- [🧠 Intelligent Fusion](#-intelligent-fusion)
- [🎨 Frontend Features](#-frontend-features)
- [📂 Project Structure](#-project-structure)
- [🏗️ Architecture](#️-architecture)
- [📊 Model Performance](#-model-performance)
  README simplified for clarity and quick use. See https://github.com/Pubu99/Abnormal-Event-Detection-Model-8 for full project history and releases.

# Abnormal Event Detection — Quick Guide

Lightweight overview and quick start for development and testing.

## What is this

A real-time anomaly detection project combining a neural model, object detection, pose and motion analysis, and a fusion engine. It provides a FastAPI backend (WebSocket + REST) and a React frontend for live visualization.

This README focuses on the essentials: run the system locally and where to find important files.

## Quick Start (Windows PowerShell)

Prereqs: Python 3.9+ and Node.js 16+ installed. GPU/CUDA optional.

1. Clone the repo:

```powershell
git clone https://github.com/Pubu99/Abnormal-Event-Detection-Model-8.git
cd Abnormal-Event-Detection-Model-8
```

2. Backend (FastAPI + PyTorch):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# If you have an NVIDIA GPU and want CUDA builds, install PyTorch per https://pytorch.org
cd ..
# Start backend (run from repo root so imports work)
python backend/api/app.py
```

API: http://localhost:8000 — Swagger: http://localhost:8000/docs

3. Frontend (React): open a new terminal

```powershell
cd frontend
npm install
npm start
```

Frontend: http://localhost:3000

## Important paths

- `backend/` — FastAPI app and backend services
- `frontend/` — React app
- `models/` — Trained model weights (keep large files here)
- `inference/` — Inference engine
- `src/` — training and research scripts
- `configs/` and `data/` — configuration and datasets

## Notes about docs and large files

- The repository `.gitignore` intentionally excludes the `docs/` folder to avoid committing large documentation and binaries. If you need `docs/` content, either download the release assets or ask the maintainer.
- Large model files should live in `models/`. For collaborative tracking of large binaries, use Git LFS or host them as release assets.

To untrack `docs/` locally (PowerShell):

```powershell
git rm -r --cached docs
git commit -m "Stop tracking docs/"
git push
```

## Quick troubleshooting

- If imports fail when starting the backend, run it from the project root (important for relative imports).
- If GPU is not detected, install CPU-only PyTorch or follow the official PyTorch install for CUDA that matches your GPU.

## Want the full documentation?

Full technical docs are available in the `docs/` folder in the project releases and contain detailed architecture, training methodology, and results. They are intentionally kept out of the main git history to reduce repo size.

---

If you'd like I can:

- Add a short quickstart script (`scripts/quick_start.ps1`) to automate the steps above.
- Produce a short `README_DEV.md` with developer notes and where to find key code (fusion engine, model entry points).

Please tell me which you'd prefer and I'll add it.

### 3. Frame Timeline (100 frames)

Visual history of recent predictions:

- **Bar Chart** showing last 100 frames
- **Color-Coded Bars** matching severity levels
- **Hover Details** showing:
  - Frame number
  - Timestamp
  - Prediction class
  - Confidence score
- **Smooth Scrolling** through timeline

### 4. Auto-Screenshot System

Automatic evidence capture for critical events:

- **Auto-Save** screenshots when confidence > 70%
- **Last 50 Screenshots** displayed in grid
- **Metadata Overlay:**
  - Timestamp
  - Prediction class
  - Confidence percentage
- **One-Click Download** without intrusive prompts
- **Thumbnail Grid** with smooth scrolling

---

## 📂 Project Structure

````
Abnormal-Event-Detection-Model-8/
│
├── backend/                          # FastAPI Backend (Port 8000)
│   ├── api/
│   │   ├── app.py                   # Main API server with WebSocket
*** Begin Minimal README ***

# Abnormal Event Detection

Simple quick-start and essential notes.

## Quick start (Windows PowerShell)

1) Clone:

```powershell
git clone https://github.com/Pubu99/Abnormal-Event-Detection-Model-8.git
cd Abnormal-Event-Detection-Model-8
````

2. Backend (from repo root):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
python backend/api/app.py
```

3. Frontend (new terminal):

```powershell
cd frontend
npm install
npm start
```

## Main folders

- `backend/` — FastAPI server
- `frontend/` — React app
- `models/` — trained weights (put large files here)

## Notes

- `docs/` is intentionally excluded from commits to keep the repo small. Get full docs from releases.
- Run backend from project root to avoid import errors.

**_ End Minimal README _**

- `GET /` - Health check
- `GET /health` - Detailed system status
- `POST /api/predict` - Upload video for batch analysis
- `POST /api/analyze-frame` - Single frame analysis
- `WS /ws/stream` - WebSocket real-time streaming
- `GET /api/classes` - List of anomaly classes
- `GET /api/detections/history` - Detection history
- `GET /api/detections/statistics` - System statistics

### 📁 Project Structure Reference

```
Abnormal-Event-Detection-Model-8/
│
├── README.md                          # This file - project overview
│
├── backend/                           # FastAPI Backend (Port 8000)
│   ├── api/
│   │   ├── app.py                    # Main FastAPI application
│   │   ├── yolov10s.pt               # YOLOv10 model weights
│   │   └── yolov8n.pt                # YOLOv8 model weights (fallback)
│   ├── core/
│   │   └── unified_pipeline.py       # Multi-modal orchestration
│   ├── services/
│   │   ├── intelligent_fusion.py     # Fusion engine (weighted voting)
│   │   ├── motion_analysis.py        # Optical Flow + MOG2
│   │   ├── pose_estimation.py        # MediaPipe pose detection
│   │   ├── object_tracking.py        # Centroid tracker
│   │   ├── rule_engine.py            # Context-aware rules (8 rules)
│   │   └── zone_manager.py           # Spatial zone configuration
│   ├── requirements.txt              # Backend Python dependencies
│   ├── README.md                     # Backend setup guide
│   └── QUICK_START.md                # One-command backend setup
│
├── frontend/                          # React Frontend (Port 3000)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ProfessionalDashboardV2.js  # Main dashboard
│   │   │   ├── LiveCameraV2.js             # Live detection UI
│   │   │   ├── AlertFeedV2.js              # Alert notifications
│   │   │   └── StatsPanel.js               # Statistics display
│   │   ├── App.js                    # Main React app
│   │   └── index.js                  # Entry point
│   ├── public/
│   │   └── index.html
│   ├── package.json                  # Node.js dependencies
│   └── README.md                     # Frontend documentation
│
├── models/
│   ├── best_model.pth                # Trained model weights (14.97M params)
│   ├── README.md                     # Model information
│   └── openpose/                     # OpenPose models (if used)
│
├── inference/
│   └── engine.py                     # Inference engine (AnomalyDetector class)
│
├── src/                              # Training/Research Code
│   ├── models/
│   │   ├── research_model.py         # Main model architecture
│   │   ├── vae.py                    # VAE for unsupervised learning
│   │   └── losses.py                 # Custom losses (Focal, MIL)
│   ├── training/
│   │   ├── research_trainer.py       # Multi-task trainer
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── sam_optimizer.py          # SAM optimizer
│   ├── data/
│   │   ├── dataset.py                # UCF Crime dataset loader
│   │   └── sequence_dataset.py       # Sequence data preparation
│   └── utils/
│       ├── logger.py                 # Training logger
│       ├── helpers.py                # Utility functions
│       └── config.py                 # Configuration management
│
├── configs/
│   ├── config_research_enhanced.yaml # Main training configuration
│   ├── config_optimized.yaml         # Optimized inference config
│   └── config.yaml                   # Base configuration
│
├── data/
│   ├── raw/                          # UCF Crime Dataset
│   │   ├── Train/                    # Training videos (1,220 clips)
│   │   └── Test/                     # Test videos (390 clips)
│   ├── processed/                    # Preprocessed frames
│   └── annotations/                  # Annotation files
│
├── docs/                             # Documentation (30,500+ words)
│   ├── TECHNICAL_OVERVIEW.md         # Complete technical overview
│   ├── ARCHITECTURE_DETAILS.md       # Architecture deep dive
│   ├── RESULTS_AND_ANALYSIS.md       # Performance analysis
│   ├── TRAINING_METHODOLOGY.md       # Training details
│   ├── ANALYSIS_VALIDATION.md        # Validation methodology
│   ├── IMPLEMENTATION_COMPLETE.md    # Implementation status
│   ├── DOCUMENTATION_SUMMARY.md      # Documentation index
│   └── NEW/                          # Latest documentation
│       ├── PROFESSIONAL_FUSION_SYSTEM.md
│       ├── LIVE_DETECTION_GUIDE.md
│       ├── QUICK_START_ENHANCED.md
│       ├── ENHANCED_SYSTEM_GUIDE.md
│       ├── RULES_GUIDE.md
│       ├── FINAL_STATUS.md
│       └── OPTIMIZATION_COMPLETE.md
│
├── outputs/                          # Training outputs
│   ├── checkpoints/                  # Model checkpoints
│   ├── logs/                         # Training logs
│   ├── models/                       # Saved models
│   ├── results/                      # Evaluation results
│   └── visualizations/               # Training visualizations
│
├── scripts/                          # Utility scripts
│   ├── download_data.py              # Dataset downloader
│   └── preprocess.py                 # Data preprocessing
│
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter notebooks (analysis)
├── uploads/                          # Uploaded videos/screenshots
│
├── train_research.py                 # Main training script
├── evaluate_research.py              # Evaluation script
├── test_setup.py                     # Setup verification
├── requirements.txt                  # Project-wide dependencies
└── .gitignore                        # Git ignore file
```

### 🎓 Academic References

This system is based on state-of-the-art research:

1. **Tan, M., & Le, Q. (2019)**. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. **Lin, T. Y., et al. (2017)**. Focal Loss for Dense Object Detection. ICCV.
3. **Shaw, P., et al. (2018)**. Self-Attention with Relative Position Representations. NAACL.
4. **Sultani, W., et al. (2018)**. Real-world Anomaly Detection in Surveillance Videos. CVPR. (UCF Crime Dataset)
5. **Redmon, J., & Farhadi, A. (2018)**. YOLOv3: An Incremental Improvement. arXiv.
6. **Bazarevsky, V., et al. (2020)**. BlazePose: On-device Real-time Body Pose tracking. arXiv. (MediaPipe)

---

## 🔬 Training the Model

### Prerequisites for Training

**Requirements:**

- Python: 3.9+
- PyTorch: 2.7.0 with CUDA support
- RAM: 16GB+ recommended
- GPU: NVIDIA GPU with 8GB+ VRAM (better GPUs reduce training time significantly)
- Storage: 800GB+ for UCF Crime dataset
- Time: ~2-3 hours per training run (with good GPU)

### Training Steps Overview

#### Step 1: Download UCF Crime Dataset

The UCF Crime dataset contains 1,610 videos (~800GB) across 14 categories. Visit the official website or use the provided download script to obtain the complete dataset with all 14 anomaly categories plus NormalVideos.

**Dataset Structure:**

- Train folder: 14 class folders (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, NormalVideos)
- Test folder: Same 14 class folders

#### Step 2: Preprocess Dataset (Optional)

The preprocessing script extracts frames from videos at 2 fps (configurable) for faster training. This saves time during the training process by avoiding repeated video decoding. Processed frames are saved to the data/processed directory.

#### Step 3: Configure Training

Edit the configuration file at `configs/config_research_enhanced.yaml` to customize training parameters. Key configurations include:

- Training parameters: epochs (100), batch size (64), learning rate (0.0001), gradient accumulation steps (2)
- Model backbone: EfficientNet-B0 (options: b0-b3), pretrained on ImageNet
- Temporal components: BiLSTM hidden dimensions (256), layers (2), Transformer layers (2), attention heads (8)
- Data settings: sequence length (16 frames), input resolution (224x224)

#### Step 4: Start Training

First verify your setup using the test script, then start training with the research-enhanced configuration. The training process will load the dataset, create frame sequences, initialize the model with pretrained EfficientNet weights, train with multi-task learning, save checkpoints to the outputs directory, and display real-time progress with metrics.

The training typically shows progressive improvement over epochs, with the model achieving high accuracy (98%+) within 15-20 epochs. The best model is automatically saved based on validation performance.

#### Step 5: Monitor Training

**TensorBoard Monitoring (Recommended):**
Launch TensorBoard in a separate terminal pointing to the outputs/logs directory. Access the web interface at localhost:6006 to view loss curves, accuracy metrics, learning rate schedules, and confusion matrices in real-time.

**Weights & Biases (Optional):**
If configured, W&B automatically tracks training metrics and provides cloud-based visualization and experiment tracking.

**Console Output:**
The training script displays rich console output with real-time progress, and all metrics are logged to outputs/logs/training.log for later review.

#### Step 6: Evaluate Model

Use the evaluation script to test the best trained model on the test set. This provides detailed performance metrics including:

# Generate detailed analysis

python evaluate_research.py --model outputs/checkpoints/best.pth --detailed

The evaluation will output detailed metrics including test accuracy (99.38%), precision, recall, and F1-scores for all classes. Per-class results show F1-scores for each category (e.g., NormalVideos: 99.53%, Stealing: 99.62%, Assault: 99.53%, etc.).

#### Step 7: Create Visualizations

Use the visualization script to generate comprehensive performance visualizations. Outputs are saved to outputs/visualizations/ and include confusion matrices, ROC curves, PR curves, class-wise performance charts, and training history plots.

#### Step 8: Deploy Model

Copy the best trained model from outputs/checkpoints/best.pth to models/best_model.pth for deployment. Verify the model loads correctly using the AnomalyDetector class, then start the backend server to use the newly trained model for inference.

### Training Tips and Best Practices

**GPU Memory Management:**
If encountering out-of-memory errors, reduce the batch size (e.g., to 32 or 16) and increase gradient accumulation steps (to 4) to maintain the effective batch size.

**Handling Overfitting:**
Increase regularization by adjusting dropout rates (e.g., to 0.6) and augmentation parameters like random erasing probability (e.g., to 0.5).

**Addressing Slow Convergence:**
Adjust the learning rate schedule by increasing the initial learning rate (e.g., to 0.0002) or reducing warmup epochs (e.g., to 5).

**Tackling Class Imbalance:**
Fine-tune the Focal Loss parameters by increasing gamma (e.g., to 3.0) for harder focus on difficult examples, and use auto-computed class weights.

### Alternative Training Options

The project includes several training scripts for different purposes:

- Standard training without research enhancements using the base configuration
- Setup validation script to verify environment before training
- Model component testing scripts for individual module validation
- Integration testing for full pipeline verification

### Resume Training

Training can be resumed from a saved checkpoint if interrupted, maintaining all optimizer states and learning rate schedules. Models can also be fine-tuned from pretrained weights by loading a checkpoint as the starting point.

For complete training methodology and hyperparameter details, see [TRAINING_METHODOLOGY.md](docs/TRAINING_METHODOLOGY.md)

---

## 🛠️ Configuration

### Backend Configuration

The main configuration file `config.yaml` controls:

- Model settings: path to weights, device (cuda/cpu), sequence length
- Fusion engine: detection weights (ML: 0.40, YOLO: 0.25, Pose: 0.20, Motion: 0.15)
- Thresholds: normal (0.3), suspicious (0.5), abnormal (0.7)
- Server settings: host, port, reload options

### Frontend Configuration

Create a `.env` file in the frontend directory to configure:

- Backend API URL (default: http://localhost:8000)
- WebSocket URL (default: ws://localhost:8000/ws/stream)
- Screenshot storage limit
- Timeline display length

---

## 🐛 Troubleshooting

### Backend Issues

**ModuleNotFoundError:**
Ensure the virtual environment is activated and all dependencies are installed via the backend requirements file.

**CUDA Out of Memory:**
Either reduce batch size in the configuration or switch to CPU mode by changing the device setting.

**WebSocket Connection Failed:**
Verify the backend is running on the correct port (8000) and check for port conflicts.

### Frontend Issues

**Cannot Connect to Backend:**
Verify the backend URL in the frontend .env file matches where the backend is running.

**Camera Not Working:**
Check browser permissions for camera access in browser settings (e.g., Chrome Settings → Privacy → Camera).

**npm Install Fails:**
Clear the npm cache and delete node_modules and package-lock.json, then reinstall dependencies.

---

## 🤝 Contributing

Contributions are welcome! This project is open for improvements and extensions.

### How to Contribute

1. Fork the repository and clone it locally
2. Create a feature branch for your changes
3. Implement your improvements (new detection modalities, fusion enhancements, UI improvements, bug fixes, documentation updates)
4. Test your changes using the provided test scripts
5. Commit with clear, descriptive messages
6. Push to your fork and open a Pull Request with a clear description

### Contribution Priority Areas

**High Priority:**

- Mobile app development (iOS/Android)
- Cloud deployment solutions (AWS/Azure/GCP)
- Multi-camera support and synchronization
- Database integration (PostgreSQL/MongoDB)

**Medium Priority:**

- Advanced rule customization UI
- Historical analytics dashboard
- Email/SMS alerting system
- Export to ONVIF standard

**Low Priority:**

- Additional data augmentation techniques
- Model compression for edge deployment
- Alternative backbone architectures
- Cross-dataset evaluation

### Code Style Guidelines

**Python (Backend):**

- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings to functions/classes
- Maximum line length: 100 characters

**JavaScript/React (Frontend):**

- Use functional components with hooks
- Follow Airbnb React style guide
- Use meaningful variable names
- Add JSDoc comments for complex functions

**Documentation**:

- Update README.md for new features
- Add technical details to docs/
- Include code examples where relevant

### Bug Reports

Found a bug? Please open an issue with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Error messages/screenshots

### Feature Requests

Have an idea? Open an issue with:

- Clear description of the feature
- Use case/motivation
- Proposed implementation (if any)
- Potential impact on existing features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Research & Datasets

- **UCF Crime Dataset** - Waqas Sultani, Chen Chen, and Mubarak Shah for providing the comprehensive real-world anomaly detection dataset
- **COCO Dataset** - For YOLO pre-training and evaluation
- **ImageNet** - For EfficientNet pre-training

### Frameworks & Libraries

- **PyTorch Team** - For the excellent deep learning framework
- **FastAPI** - For modern, fast Python web framework
- **React Team** - For powerful frontend library
- **Ultralytics** - For YOLO implementation and object tracking
- **Google MediaPipe** - For real-time pose estimation
- **OpenCV** - For computer vision utilities

### Research Papers

- **EfficientNet**: Mingxing Tan and Quoc V. Le (Google Brain)
- **Focal Loss**: Tsung-Yi Lin et al. (Facebook AI Research)
- **Transformer**: Ashish Vaswani et al. (Google Brain)
- **LSTM**: Sepp Hochreiter and Jürgen Schmidhuber
- **YOLO**: Joseph Redmon and Ali Farhadi

### Special Thanks

- Academic supervisors for guidance and support throughout this Final Year Project
- University for providing resources and infrastructure
- Open Source Community for tools, libraries, and inspiration

---

## 📧 Contact & Support

### Project Information

- **Repository**: [https://github.com/Pubu99/Abnormal-Event-Detection-Model-8](https://github.com/Pubu99/Abnormal-Event-Detection-Model-8)
- **Documentation**: See `docs/` folder for 30,500+ words of technical documentation
- **Issues**: [GitHub Issues](https://github.com/Pubu99/Abnormal-Event-Detection-Model-8/issues)

### Author

- **GitHub**: [@Pubu99](https://github.com/Pubu99)
- **Project**: Final Year Project - Advanced AI/ML Research

### Getting Help

1. **Documentation**: Check `docs/` folder first
2. **Common Issues**: See Troubleshooting section above
3. **GitHub Issues**: For bugs and feature requests
4. **Discussions**: GitHub Discussions for questions

### Citation

If you use this project in your research, please cite the UCF Crime dataset paper and reference this implementation.

---

## 🎯 Project Status & Roadmap

### ✅ Completed (v3.0 - Current)

- ✅ **Core ML Model** - 99.38% test accuracy achieved
- ✅ **Multi-Modal Detection** - 6 detection services integrated
- ✅ **Intelligent Fusion** - Weighted voting with override logic
- ✅ **FastAPI Backend** - REST + WebSocket APIs
- ✅ **React Frontend** - Professional UI with real-time visualization
- ✅ **Auto-Evidence Capture** - Screenshot system with metadata
- ✅ **Comprehensive Documentation** - 30,500+ words
- ✅ **GPU Acceleration** - CUDA support
- ✅ **Object Tracking** - Centroid-based tracker with persistent IDs
- ✅ **Context-Aware Rules** - 8 intelligent rules for alerting

### 🔄 In Progress

- 🔄 Performance optimization for edge devices
- 🔄 Extended test coverage
- 🔄 Docker containerization
- 🔄 CI/CD pipeline setup

### �️ Future Roadmap

**Short-term:**

- Mobile application development
- Database integration
- User authentication & authorization
- Multi-language support

**Medium-term:**

- Cloud deployment solutions
- Multi-camera synchronized detection
- Historical data analytics
- Advanced reporting dashboard
- Email/SMS alert integration

**Long-term:**

- Edge deployment support
- Cross-dataset evaluation
- Model compression techniques
- Real-time 4K video support
- Integration with existing CCTV systems
- Federated learning for privacy-preserving training

---

## 📊 Project Highlights

**Project Metrics:**

- Lines of Code: 15,000+
- Documentation: 30,500+ words
- Test Accuracy: 99.38%
- Model Parameters: 14,966,922
- Detection Modalities: 6
- API Endpoints: 12+
- Anomaly Classes: 14
- Real-time FPS: 30-35
- Development Time: 6 months
- Status: Open for contributions

---

<div align="center">

**⭐ Star this repository if you find it useful!**

**🔔 Watch for updates and new features**

**🍴 Fork to contribute your improvements**

---

**Built for Professional Video Surveillance & Public Safety**

_Making the world safer through AI-powered anomaly detection_

---

[![GitHub stars](https://img.shields.io/github/stars/Pubu99/Abnormal-Event-Detection-Model-8?style=social)](https://github.com/Pubu99/Abnormal-Event-Detection-Model-8/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Pubu99/Abnormal-Event-Detection-Model-8?style=social)](https://github.com/Pubu99/Abnormal-Event-Detection-Model-8/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/Pubu99/Abnormal-Event-Detection-Model-8?style=social)](https://github.com/Pubu99/Abnormal-Event-Detection-Model-8/watchers)

</div>
