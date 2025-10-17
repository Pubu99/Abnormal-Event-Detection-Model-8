# Professional Fusion System - Frontend Startup Script
# Start the React frontend for anomaly detection display

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   STARTING PROFESSIONAL FUSION FRONTEND" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Features:" -ForegroundColor Yellow
Write-Host "   [OK] Anomaly-Only Display (No Normal highlights)"
Write-Host "   [OK] Multi-Modal Score Breakdown"
Write-Host "   [OK] Real-time Fusion Reasoning"
Write-Host "   [OK] Detection History Timeline"
Write-Host "   [OK] Severity-Based Alerts"
Write-Host ""

Write-Host "Starting React dev server on http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "[WARNING] Make sure backend is running on port 8000!" -ForegroundColor Yellow
Write-Host "   Run: .\START_BACKEND.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Navigate to frontend and start
cd frontend
npm start
