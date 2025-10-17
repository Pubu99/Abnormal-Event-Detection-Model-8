import React from "react";

function ResultsDisplay({ result }) {
  if (!result) return null;

  const getAnomalyColor = () => {
    if (result.dangerous_objects && result.dangerous_objects.length > 0) {
      return "bg-red-600";
    }
    if (result.anomaly_detected) {
      return "bg-orange-600";
    }
    return "bg-green-600";
  };

  const getAnomalyIcon = () => {
    if (result.dangerous_objects && result.dangerous_objects.length > 0) {
      return "🚨";
    }
    if (result.anomaly_detected) {
      return "⚠️";
    }
    return "✅";
  };

  const getAnomalyMessage = () => {
    if (result.dangerous_objects && result.dangerous_objects.length > 0) {
      return "CRITICAL: Dangerous Objects Detected!";
    }
    if (result.anomaly_detected) {
      return "Anomaly Detected";
    }
    return "No Anomaly Detected";
  };

  return (
    <div className="bg-gray-800 rounded-lg shadow-xl p-4 sm:p-6 animate-fadeIn">
      <h2 className="text-xl sm:text-2xl font-bold text-white mb-4 sm:mb-6 flex items-center">
        <svg
          className="w-6 h-6 sm:w-8 sm:h-8 mr-2 sm:mr-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
        </svg>
        Analysis Results
      </h2>

      {/* Status Banner */}
      <div
        className={`${getAnomalyColor()} rounded-lg p-4 sm:p-6 mb-4 sm:mb-6`}
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between">
          <div className="mb-3 sm:mb-0">
            <h3 className="text-white text-lg sm:text-2xl font-bold mb-1 sm:mb-2">
              {getAnomalyIcon()} {getAnomalyMessage()}
            </h3>
            <p className="text-white text-opacity-90 text-sm sm:text-base">
              Detected:{" "}
              <span className="font-semibold">{result.predicted_class}</span>
            </p>
          </div>
          <div className="text-left sm:text-right">
            <p className="text-white text-opacity-75 text-xs sm:text-sm">
              Confidence
            </p>
            <p className="text-white text-2xl sm:text-4xl font-bold">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Primary Prediction */}
        <div className="bg-gray-700 rounded-lg p-4 sm:p-6">
          <h3 className="text-white font-semibold mb-3 sm:mb-4 flex items-center text-sm sm:text-base">
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
              <path
                fillRule="evenodd"
                d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                clipRule="evenodd"
              />
            </svg>
            Detection Details
          </h3>
          <div className="space-y-3">
            <div>
              <p className="text-gray-400 text-xs sm:text-sm">
                Predicted Event
              </p>
              <p className="text-white text-lg sm:text-xl font-bold">
                {result.predicted_class}
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-xs sm:text-sm">Anomaly Score</p>
              <div className="w-full bg-gray-600 rounded-full h-2 sm:h-3 mt-1">
                <div
                  className="bg-blue-500 h-2 sm:h-3 rounded-full transition-all"
                  style={{ width: `${result.max_anomaly_score * 100}%` }}
                ></div>
              </div>
              <p className="text-white text-xs sm:text-sm mt-1">
                {(result.max_anomaly_score * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Top 3 Predictions */}
        <div className="bg-gray-700 rounded-lg p-4 sm:p-6">
          <h3 className="text-white font-semibold mb-3 sm:mb-4 flex items-center text-sm sm:text-base">
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
            </svg>
            Top 3 Possibilities
          </h3>
          <div className="space-y-3">
            {result.top3_predictions?.map((pred, index) => (
              <div key={index} className="bg-gray-600 rounded-lg p-3">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold">
                    {index + 1}. {pred.class}
                  </span>
                  <span className="text-blue-400 font-bold">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      index === 0
                        ? "bg-blue-500"
                        : index === 1
                        ? "bg-purple-500"
                        : "bg-indigo-500"
                    }`}
                    style={{ width: `${pred.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Dangerous Objects Alert */}
      {result.dangerous_objects && result.dangerous_objects.length > 0 && (
        <div className="mt-6 bg-red-900 bg-opacity-50 border-2 border-red-500 rounded-lg p-6">
          <h3 className="text-red-300 font-bold text-lg mb-4 flex items-center">
            <svg
              className="w-6 h-6 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            ⚠️ DANGEROUS OBJECTS DETECTED
          </h3>
          <div className="space-y-2">
            {result.dangerous_objects.map((obj, index) => (
              <div key={index} className="bg-red-800 bg-opacity-50 rounded p-3">
                <p className="text-white">
                  <span className="font-semibold">
                    Frame {obj.frame_index}:
                  </span>{" "}
                  {obj.objects.join(", ")}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="mt-6 flex space-x-4">
        <button className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors">
          📊 View Detailed Report
        </button>
        <button className="flex-1 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 rounded-lg transition-colors">
          📤 Export Results
        </button>
        <button className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition-colors">
          ✅ Mark as Verified
        </button>
      </div>
    </div>
  );
}

export default ResultsDisplay;
