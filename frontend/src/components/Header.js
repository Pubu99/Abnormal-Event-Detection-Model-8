import React from "react";

function Header() {
  return (
    <header className="bg-gray-800 shadow-lg border-b-2 border-blue-500">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 sm:space-x-4">
            <div className="bg-blue-500 p-2 sm:p-3 rounded-lg">
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
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-lg sm:text-2xl font-bold text-white">
                Anomaly Detection System
              </h1>
              <p className="text-gray-400 text-xs sm:text-sm hidden sm:block">
                Multi-Camera Surveillance with AI
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 sm:space-x-4">
            <div className="flex items-center space-x-1 sm:space-x-2 bg-green-900 bg-opacity-50 px-2 sm:px-4 py-1 sm:py-2 rounded-lg">
              <div className="w-2 h-2 sm:w-3 sm:h-3 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 font-semibold text-xs sm:text-base">
                <span className="hidden sm:inline">System Active</span>
                <span className="inline sm:hidden">Active</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
