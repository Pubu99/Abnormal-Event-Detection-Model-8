import React, { useState } from "react";
import "./App.css";
import VideoUpload from "./components/VideoUpload";
import LiveCamera from "./components/LiveCamera";
import ResultsDisplay from "./components/ResultsDisplay";
import Header from "./components/Header";
import Stats from "./components/Stats";

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      <Header />

      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 lg:py-8 max-w-7xl">
        {/* Stats Overview */}
        <Stats />

        {/* Tab Navigation */}
        <div className="bg-gray-800 rounded-lg shadow-xl p-4 sm:p-6 mb-4 sm:mb-6">
          <div className="flex flex-col sm:flex-row sm:space-x-4 space-y-2 sm:space-y-0 border-b border-gray-700 mb-4 sm:mb-6">
            <button
              onClick={() => setActiveTab("upload")}
              className={`px-4 sm:px-6 py-2 sm:py-3 font-semibold transition-all rounded-t-lg sm:rounded-none ${
                activeTab === "upload"
                  ? "text-blue-400 border-b-2 border-blue-400 bg-gray-700 sm:bg-transparent"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-700"
              }`}
            >
              <span className="inline sm:hidden">📹 Upload</span>
              <span className="hidden sm:inline">📹 Video Upload</span>
            </button>
            <button
              onClick={() => setActiveTab("live")}
              className={`px-4 sm:px-6 py-2 sm:py-3 font-semibold transition-all rounded-t-lg sm:rounded-none ${
                activeTab === "live"
                  ? "text-blue-400 border-b-2 border-blue-400 bg-gray-700 sm:bg-transparent"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-700"
              }`}
            >
              <span className="inline sm:hidden">🔴 Live</span>
              <span className="hidden sm:inline">🔴 Live Camera</span>
            </button>
          </div>

          {/* Content */}
          {activeTab === "upload" && (
            <VideoUpload
              onAnalysisComplete={handleAnalysisComplete}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          )}

          {activeTab === "live" && (
            <LiveCamera onDetection={handleAnalysisComplete} />
          )}
        </div>

        {/* Results Display */}
        {analysisResult && <ResultsDisplay result={analysisResult} />}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>🎓 FYP 2025 - Abnormal Event Detection System</p>
          <p className="text-sm mt-2">
            Powered by EfficientNet + BiLSTM + Transformer | 99.38% Accuracy
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
