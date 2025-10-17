import React, { useState, useRef } from "react";
import axios from "axios";

function VideoUpload({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  const API_URL = "http://localhost:8000";

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Validate file type
    const validTypes = [
      "video/mp4",
      "video/avi",
      "video/mov",
      "video/x-matroska",
    ];
    if (!validTypes.includes(file.type)) {
      alert("Please upload a valid video file (mp4, avi, mov, mkv)");
      return;
    }

    // Validate file size (max 500MB)
    if (file.size > 500 * 1024 * 1024) {
      alert("File size must be less than 500MB");
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a video file first");
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setProgress(percentCompleted);
        },
      });

      onAnalysisComplete(response.data);
    } catch (error) {
      console.error("Error uploading video:", error);
      alert(
        "Failed to analyze video. Make sure the backend server is running."
      );
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all ${
          dragActive
            ? "border-blue-500 bg-blue-500 bg-opacity-10"
            : "border-gray-600 hover:border-gray-500"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleChange}
          className="hidden"
        />

        {!selectedFile ? (
          <div className="space-y-4">
            <div className="flex justify-center">
              <svg
                className="w-20 h-20 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <div>
              <p className="text-xl text-gray-300 mb-2">
                Drag & drop your video here
              </p>
              <p className="text-gray-500 mb-4">
                or click to browse (MP4, AVI, MOV, MKV - Max 500MB)
              </p>
              <button
                onClick={() => fileInputRef.current.click()}
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors"
              >
                Select Video File
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-center">
              <svg
                className="w-16 h-16 text-green-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div>
              <p className="text-lg text-gray-300 font-semibold">
                {selectedFile.name}
              </p>
              <p className="text-gray-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={() => setSelectedFile(null)}
              className="text-red-400 hover:text-red-300 text-sm"
            >
              Remove file
            </button>
          </div>
        )}
      </div>

      {/* Upload Button */}
      {selectedFile && (
        <button
          onClick={handleUpload}
          disabled={isAnalyzing}
          className={`w-full py-4 rounded-lg font-semibold text-lg transition-all ${
            isAnalyzing
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
          }`}
        >
          {isAnalyzing ? (
            <div className="flex items-center justify-center space-x-3">
              <svg
                className="animate-spin h-6 w-6 text-white"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>Analyzing Video... {progress}%</span>
            </div>
          ) : (
            "🚀 Analyze Video for Anomalies"
          )}
        </button>
      )}

      {/* Info Panel */}
      <div className="bg-blue-900 bg-opacity-30 border border-blue-500 rounded-lg p-4">
        <h3 className="text-blue-300 font-semibold mb-2 flex items-center">
          <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
          Detection Capabilities
        </h3>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>
            ✓ 14 Anomaly Types: Abuse, Arrest, Arson, Assault, Burglary, etc.
          </li>
          <li>✓ Object Detection: Guns, knives, fire, suspicious objects</li>
          <li>✓ Confidence Scoring: Get probability for each prediction</li>
          <li>✓ Real-time Processing: Fast inference with GPU acceleration</li>
        </ul>
      </div>
    </div>
  );
}

export default VideoUpload;
