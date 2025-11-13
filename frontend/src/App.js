import React, { useState } from "react";
import "./App.css";
import "./styles/professional.css";
import ProfessionalDashboardV2 from "./components/ProfessionalDashboardV2";
import MultiCameraDashboard from "./components/MultiCameraDashboard";
import AdminPanel from "./components/AdminPanel";
import AnomalyHistory from "./components/AnomalyHistory";

function App() {
  const [showAdmin, setShowAdmin] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [viewMode, setViewMode] = useState("single"); // "single" or "multi"
  
  return (
    <div className="min-h-screen bg-slate-950">
      {/* Professional Header */}
      <header className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700 shadow-2xl">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/50">
                <svg
                  className="w-7 h-7 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">
                  ANOMALY DETECTION SYSTEM
                </h1>
                <p className="text-slate-400 text-sm">
                  Enterprise Surveillance Intelligence Platform
                </p>
              </div>
            </div>
            {/* System Status Indicator */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-slate-400 text-sm">System Online</span>
              </div>
              {/* View Mode Toggle */}
              <button
                onClick={() => setViewMode(viewMode === "single" ? "multi" : "single")}
                className="ml-4 px-3 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition"
                title={viewMode === "single" ? "Switch to Multi-Camera Grid" : "Switch to Single Camera"}
              >
                {viewMode === "single" ? "📹 Multi-Camera Grid" : "🎥 Single Camera"}
              </button>
              {/* Admin & History toggles */}
              <button
                onClick={() => setShowHistory(true)}
                className="ml-2 px-3 py-1 rounded bg-slate-800 text-slate-200 text-sm hover:bg-slate-700 transition"
              >
                History
              </button>
              <button
                onClick={() => setShowAdmin(true)}
                className="ml-2 px-3 py-1 rounded bg-slate-800 text-slate-200 text-sm hover:bg-slate-700 transition"
              >
                Admin
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main>
        {viewMode === "multi" ? (
          <MultiCameraDashboard />
        ) : (
          <ProfessionalDashboardV2 />
        )}
      </main>

      {showAdmin && <AdminPanel onClose={() => setShowAdmin(false)} />}
      {showHistory && <AnomalyHistory onClose={() => setShowHistory(false)} />}

      {/* Professional Footer */}
      <footer className="bg-slate-900/50 backdrop-blur-sm border-t border-slate-800 mt-8">
        <div className="px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-center md:text-left">
              <p className="text-slate-400 text-sm">
                🎓 Final Year Project 2025 - Advanced AI/ML Research
              </p>
              <p className="text-slate-500 text-xs mt-1">
                EfficientNet-B0 + BiLSTM + Transformer | Multi-Modal Fusion
                Engine
              </p>
            </div>
            <div className="flex items-center gap-6 text-xs text-slate-500">
              <span>UCF Crime Dataset</span>
              <span>•</span>
              <span>14 Anomaly Classes</span>
              <span>•</span>
              <span>Real-Time Processing</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;