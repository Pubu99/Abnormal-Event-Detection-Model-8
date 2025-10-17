import React, { useState, useEffect } from "react";

function Stats() {
  const [stats, setStats] = useState({
    modelAccuracy: "99.38%",
    processingSpeed: "2.6h",
    classesSupported: 14,
    status: "Ready",
  });

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-6">
      {/* Model Accuracy */}
      <div className="bg-gradient-to-br from-blue-600 to-blue-800 rounded-lg p-4 sm:p-6 shadow-xl">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-blue-200 text-xs sm:text-sm font-semibold">
              Model Accuracy
            </p>
            <p className="text-white text-xl sm:text-3xl font-bold mt-1 sm:mt-2">
              {stats.modelAccuracy}
            </p>
          </div>
          <div className="bg-blue-500 bg-opacity-30 p-2 sm:p-3 rounded-full">
            <svg
              className="w-6 h-6 sm:w-8 sm:h-8 text-white"
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
        </div>
      </div>

      {/* Training Speed */}
      <div className="bg-gradient-to-br from-green-600 to-green-800 rounded-lg p-4 sm:p-6 shadow-xl">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-green-200 text-xs sm:text-sm font-semibold">
              Training Time
            </p>
            <p className="text-white text-xl sm:text-3xl font-bold mt-1 sm:mt-2">
              {stats.processingSpeed}
            </p>
          </div>
          <div className="bg-green-500 bg-opacity-30 p-2 sm:p-3 rounded-full">
            <svg
              className="w-6 h-6 sm:w-8 sm:h-8 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Classes Supported */}
      <div className="bg-gradient-to-br from-purple-600 to-purple-800 rounded-lg p-4 sm:p-6 shadow-xl">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-purple-200 text-xs sm:text-sm font-semibold">
              Anomaly Classes
            </p>
            <p className="text-white text-xl sm:text-3xl font-bold mt-1 sm:mt-2">
              {stats.classesSupported}
            </p>
          </div>
          <div className="bg-purple-500 bg-opacity-30 p-2 sm:p-3 rounded-full">
            <svg
              className="w-6 h-6 sm:w-8 sm:h-8 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-gradient-to-br from-orange-600 to-orange-800 rounded-lg p-4 sm:p-6 shadow-xl col-span-2 lg:col-span-1">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-orange-200 text-xs sm:text-sm font-semibold">
              System Status
            </p>
            <p className="text-white text-xl sm:text-3xl font-bold mt-1 sm:mt-2">
              {stats.status}
            </p>
          </div>
          <div className="bg-orange-500 bg-opacity-30 p-2 sm:p-3 rounded-full">
            <svg
              className="w-6 h-6 sm:w-8 sm:h-8 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Stats;
