# Professional Fusion System - Backend Startup Script
# Start the enhanced API with Intelligent Fusion Engine

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   STARTING PROFESSIONAL FUSION BACKEND" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Features:" -ForegroundColor Yellow
Write-Host "   [OK] Weighted Fusion (ML 40%, YOLO 25%, Pose 20%, Motion 15%)"
Write-Host "   [OK] Anomaly-Only Detection (threshold 0.70)"
Write-Host "   [OK] Person Falling Detection (even if ML says Normal)"
Write-Host "   [OK] Detection History & Statistics"
Write-Host "   [OK] Real-time WebSocket streaming"
Write-Host ""

# Check if model exists
$modelPath = "models\best_model.pth"
if (Test-Path $modelPath) {
    Write-Host "[OK] ML Model found: $modelPath" -ForegroundColor Green
} else {
    Write-Host "[WARNING] ML Model not found at $modelPath" -ForegroundColor Red
    Write-Host "   Please ensure best_model.pth is in the models/ directory" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "WebSocket Endpoint: ws://localhost:8000/ws/stream" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Navigate to backend/api and start the server
cd backend\api
python app.py
