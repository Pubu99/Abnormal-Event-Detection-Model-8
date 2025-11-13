import React, { useState, useEffect, useCallback } from "react";

export default function CameraManager({ onClose }) {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  
  // Form state
  const [formData, setFormData] = useState({
    name: "",
    type: "webcam",
    location: "",
    rtsp_url: "",
    username: "",
    password: "",
    resolution_width: 1280,
    resolution_height: 720,
    fps: 15,
    enabled: true,
    // Network access fields
    network_access_method: "direct",
    ssh_host: "",
    ssh_port: 22,
    ssh_user: "",
    ssh_key_path: "",
    tunnel_local_port: 8554,
    proxy_host: "",
    proxy_port: 8554
    ,
    // Pose tuning defaults (backend CameraManager will accept these)
    pose_body_angle_thresh: 45.0,
    pose_velocity_thresh: 0.02,
    pose_acceleration_thresh: 0.02,
    pose_persistence_frames: 3
  });

  const apiBase = process.env.REACT_APP_API_BASE || "http://localhost:8000";

  const loadCameras = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/api/cameras`);
      const data = await response.json();
      if (data.success) {
        setCameras(data.cameras);
      }
    } catch (error) {
      console.error("Error loading cameras:", error);
      alert("Failed to load cameras");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    loadCameras();
  }, [loadCameras]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    // Normalize numeric inputs to numbers so backend receives proper types
    let newVal = value;
    if (type === 'checkbox') newVal = checked;
    else if (type === 'number') {
      // allow empty -> keep as empty string
      newVal = value === '' ? '' : Number(value);
    }
    setFormData(prev => ({
      ...prev,
      [name]: newVal
    }));
  };

  const resetForm = () => {
    setFormData({
      name: "",
      type: "webcam",
      location: "",
      rtsp_url: "",
      username: "",
      password: "",
      resolution_width: 1280,
      resolution_height: 720,
      fps: 15,
      enabled: true,
      network_access_method: "direct",
      ssh_host: "",
      ssh_port: 22,
      ssh_user: "",
      ssh_key_path: "",
      tunnel_local_port: 8554,
      proxy_host: "",
      proxy_port: 8554
      ,
      pose_body_angle_thresh: 45.0,
      pose_velocity_thresh: 0.02,
      pose_acceleration_thresh: 0.02,
      pose_persistence_frames: 3
    });
    setEditingCamera(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const url = editingCamera 
        ? `${apiBase}/api/cameras/${editingCamera.id}`
        : `${apiBase}/api/cameras`;
      
      const method = editingCamera ? "PUT" : "POST";
      
      const response = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok && data.success) {
        alert(data.message);
        loadCameras();
        setShowAddForm(false);
        resetForm();
      } else {
        alert(`Error: ${data.detail || "Failed to save camera"}`);
      }
    } catch (error) {
      console.error("Error saving camera:", error);
      alert("Failed to save camera");
    }
  };

  const handleEdit = (camera) => {
    setEditingCamera(camera);
    setFormData({
      name: camera.name,
      type: camera.type,
      location: camera.location,
      rtsp_url: camera.rtsp_url || "",
      username: camera.username || "",
      password: camera.password || "",
      resolution_width: camera.resolution_width,
      resolution_height: camera.resolution_height,
      fps: camera.fps,
      enabled: camera.enabled,
      network_access_method: camera.network_access_method || "direct",
      ssh_host: camera.ssh_host || "",
      ssh_port: camera.ssh_port || 22,
      ssh_user: camera.ssh_user || "",
      ssh_key_path: camera.ssh_key_path || "",
      tunnel_local_port: camera.tunnel_local_port || 8554,
      proxy_host: camera.proxy_host || "",
      proxy_port: camera.proxy_port || 8554
      ,
      pose_body_angle_thresh: camera.pose_body_angle_thresh || 45.0,
      pose_velocity_thresh: camera.pose_velocity_thresh || 0.02,
      pose_acceleration_thresh: camera.pose_acceleration_thresh || 0.02,
      pose_persistence_frames: camera.pose_persistence_frames || 3
    });
    setShowAddForm(true);
  };

  const handleDelete = async (cameraId) => {
    if (!window.confirm("Are you sure you want to delete this camera?")) {
      return;
    }

    try {
      const response = await fetch(`${apiBase}/api/cameras/${cameraId}`, {
        method: "DELETE"
      });

      const data = await response.json();

      if (response.ok && data.success) {
        alert(data.message);
        loadCameras();
      } else {
        alert(`Error: ${data.detail || "Failed to delete camera"}`);
      }
    } catch (error) {
      console.error("Error deleting camera:", error);
      alert("Failed to delete camera");
    }
  };

  const testConnection = async (cameraId) => {
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/api/cameras/${cameraId}/test`);
      const data = await response.json();
      
      if (data.success && data.status === "online") {
        alert(`✅ ${data.message}`);
      } else {
        alert(`❌ ${data.message}`);
      }
      
      loadCameras(); // Refresh to show updated status
    } catch (error) {
      console.error("Error testing connection:", error);
      alert("Failed to test connection");
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      online: "bg-emerald-500",
      offline: "bg-slate-600",
      connecting: "bg-yellow-500 animate-pulse",
      error: "bg-red-500"
    };
    return colors[status] || "bg-slate-600";
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-xl border border-slate-700 shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-white font-bold text-2xl">📹 Camera Management</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Add Camera Button */}
          {!showAddForm && (
            <button
              onClick={() => {
                resetForm();
                setShowAddForm(true);
              }}
              className="mb-6 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg"
            >
              <div className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Add New Camera</span>
              </div>
            </button>
          )}

          {/* Add/Edit Form */}
          {showAddForm && (
            <form onSubmit={handleSubmit} className="bg-slate-800/50 rounded-lg p-6 mb-6 border border-slate-700">
              <h3 className="text-white font-bold text-lg mb-4">
                {editingCamera ? "Edit Camera" : "Add New Camera"}
              </h3>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    Camera Name *
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    required
                    className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                    placeholder="e.g., Main Entrance Camera"
                  />
                </div>

                <div>
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    Location *
                  </label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleInputChange}
                    required
                    className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                    placeholder="e.g., Building A Entrance"
                  />
                </div>
              </div>

              <div className="mb-4">
                <label className="block text-slate-300 text-sm font-semibold mb-2">
                  Camera Type *
                </label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="type"
                      value="webcam"
                      checked={formData.type === "webcam"}
                      onChange={handleInputChange}
                      className="text-cyan-500"
                    />
                    <span className="text-white">💻 Webcam</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="type"
                      value="ip_camera"
                      checked={formData.type === "ip_camera"}
                      onChange={handleInputChange}
                      className="text-cyan-500"
                    />
                    <span className="text-white">📹 IP Camera (RTSP)</span>
                  </label>
                </div>
              </div>

              {/* IP Camera Fields */}
              {formData.type === "ip_camera" && (
                <>
                  <div className="mb-4">
                    <label className="block text-slate-300 text-sm font-semibold mb-2">
                      RTSP URL *
                    </label>
                    <input
                      type="text"
                      name="rtsp_url"
                      value={formData.rtsp_url}
                      onChange={handleInputChange}
                      required={formData.type === "ip_camera"}
                      className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500 font-mono text-sm"
                      placeholder="rtsp://192.168.1.100:554/stream"
                    />
                    <p className="text-slate-400 text-xs mt-1">
                      Example: rtsp://username:password@192.168.1.100:554/stream1
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <label className="block text-slate-300 text-sm font-semibold mb-2">
                        Username (Optional)
                      </label>
                      <input
                        type="text"
                        name="username"
                        value={formData.username}
                        onChange={handleInputChange}
                        className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-300 text-sm font-semibold mb-2">
                        Password (Optional)
                      </label>
                      <input
                        type="password"
                        name="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                  </div>

                  {/* Network Access Configuration (for cross-network cameras) */}
                  <div className="mt-6 p-4 border-t border-slate-700">
                    <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                      <span>🌉</span>
                      <span>Network Access (for cameras on different networks)</span>
                    </h4>
                    <p className="text-slate-400 text-xs mb-4">
                      Configure network bridging if camera is on a different network
                    </p>

                    <div className="mb-4">
                      <label className="block text-slate-300 text-sm font-semibold mb-2">
                        Access Method
                      </label>
                      <select
                        name="network_access_method"
                        value={formData.network_access_method || "direct"}
                        onChange={handleInputChange}
                        className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                      >
                        <option value="direct">Direct Connection (Default)</option>
                        <option value="ssh_tunnel">SSH Tunnel (Recommended)</option>
                        <option value="rtsp_proxy">RTSP Proxy/Relay</option>
                        <option value="vpn">VPN Connection</option>
                      </select>
                      <p className="text-slate-400 text-xs mt-1">
                        Choose 'SSH Tunnel' for cameras on different networks
                      </p>
                    </div>

                    {formData.network_access_method === "ssh_tunnel" && (
                      <div className="space-y-4 pl-4 border-l-2 border-cyan-500/30">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-slate-300 text-sm font-semibold mb-2">
                              SSH Gateway Host *
                            </label>
                            <input
                              type="text"
                              name="ssh_host"
                              value={formData.ssh_host || ""}
                              onChange={handleInputChange}
                              placeholder="gateway.example.com"
                              className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                              required
                            />
                            <p className="text-slate-400 text-xs mt-1">
                              Server with access to camera network
                            </p>
                          </div>
                          <div>
                            <label className="block text-slate-300 text-sm font-semibold mb-2">
                              SSH Port
                            </label>
                            <input
                              type="number"
                              name="ssh_port"
                              value={formData.ssh_port || 22}
                              onChange={handleInputChange}
                              className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-slate-300 text-sm font-semibold mb-2">
                              SSH Username *
                            </label>
                            <input
                              type="text"
                              name="ssh_user"
                              value={formData.ssh_user || ""}
                              onChange={handleInputChange}
                              placeholder="username"
                              className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                              required
                            />
                          </div>
                          <div>
                            <label className="block text-slate-300 text-sm font-semibold mb-2">
                              Local Tunnel Port
                            </label>
                            <input
                              type="number"
                              name="tunnel_local_port"
                              value={formData.tunnel_local_port || 8554}
                              onChange={handleInputChange}
                              className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block text-slate-300 text-sm font-semibold mb-2">
                            SSH Private Key Path (Optional)
                          </label>
                          <input
                            type="text"
                            name="ssh_key_path"
                            value={formData.ssh_key_path || ""}
                            onChange={handleInputChange}
                            placeholder="/home/user/.ssh/id_rsa"
                            className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                          />
                          <p className="text-slate-400 text-xs mt-1">
                            Leave empty for password authentication. Key path must be on server.
                          </p>
                        </div>

                        <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                          <p className="text-cyan-300 text-xs">
                            <strong>💡 How it works:</strong> SSH tunnel creates a secure connection:<br />
                            <code className="text-cyan-400">localhost:{formData.tunnel_local_port || 8554}</code> → 
                            <code className="text-cyan-400"> {formData.ssh_host || 'gateway'}</code> → 
                            <code className="text-cyan-400"> camera</code>
                          </p>
                        </div>
                      </div>
                    )}

                    {formData.network_access_method === "rtsp_proxy" && (
                      <div className="space-y-4 pl-4 border-l-2 border-purple-500/30">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-slate-300 text-sm font-semibold mb-2">
                              Proxy Host *
                            </label>
                            <input
                              type="text"
                              name="proxy_host"
                              value={formData.proxy_host || ""}
                              onChange={handleInputChange}
                              placeholder="proxy.example.com"
                              className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                              required
                            />
                          </div>
                          <div>
                            <label className="block text-slate-300 text-sm font-semibold mb-2">
                              Proxy Port *
                            </label>
                            <input
                              type="number"
                              name="proxy_port"
                              value={formData.proxy_port || 8554}
                              onChange={handleInputChange}
                              className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                              required
                            />
                          </div>
                        </div>
                        <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
                          <p className="text-purple-300 text-xs">
                            <strong>💡 Proxy Mode:</strong> Connects through an RTSP relay server (like mediamtx)
                          </p>
                        </div>
                      </div>
                    )}

                    {formData.network_access_method === "vpn" && (
                      <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-3">
                        <p className="text-green-300 text-xs">
                          <strong>💡 VPN Mode:</strong> Assumes VPN connection is already established. Camera will be accessed directly via RTSP URL.
                        </p>
                      </div>
                    )}
                  </div>
                </>
              )}

              {/* Settings */}
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    Width
                  </label>
                  <input
                    type="number"
                    name="resolution_width"
                    value={formData.resolution_width}
                    onChange={handleInputChange}
                    className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    Height
                  </label>
                  <input
                    type="number"
                    name="resolution_height"
                    value={formData.resolution_height}
                    onChange={handleInputChange}
                    className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    FPS
                  </label>
                  <input
                    type="number"
                    name="fps"
                    value={formData.fps}
                    onChange={handleInputChange}
                    className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              <div className="mb-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    name="enabled"
                    checked={formData.enabled}
                    onChange={handleInputChange}
                    className="w-4 h-4 text-cyan-500"
                  />
                  <span className="text-white">Enable camera</span>
                </label>
              </div>

              {/* Pose Tuning */}
              <div className="mt-4 p-4 border-t border-slate-700">
                <h4 className="text-white font-semibold mb-3">🧭 Pose Tuning (reduce false positives)</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-slate-300 text-sm font-semibold mb-2">Body Angle Threshold (deg)</label>
                    <input
                      type="number"
                      step="0.1"
                      name="pose_body_angle_thresh"
                      value={formData.pose_body_angle_thresh}
                      onChange={handleInputChange}
                      className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-300 text-sm font-semibold mb-2">Persistence Frames</label>
                    <input
                      type="number"
                      name="pose_persistence_frames"
                      value={formData.pose_persistence_frames}
                      onChange={handleInputChange}
                      className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-300 text-sm font-semibold mb-2">Velocity Threshold</label>
                    <input
                      type="number"
                      step="0.001"
                      name="pose_velocity_thresh"
                      value={formData.pose_velocity_thresh}
                      onChange={handleInputChange}
                      className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-300 text-sm font-semibold mb-2">Acceleration Threshold</label>
                    <input
                      type="number"
                      step="0.001"
                      name="pose_acceleration_thresh"
                      value={formData.pose_acceleration_thresh}
                      onChange={handleInputChange}
                      className="w-full bg-slate-700 text-white border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                </div>
                <p className="text-slate-400 text-xs mt-3">These parameters are per-camera and help reduce pose-based false positives (e.g., require body tilt & persistence).</p>
              </div>

              <div className="flex gap-3">
                <button
                  type="submit"
                  className="bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 text-white font-semibold py-2 px-6 rounded-lg transition-all"
                >
                  {editingCamera ? "Update Camera" : "Add Camera"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowAddForm(false);
                    resetForm();
                  }}
                  className="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-2 px-6 rounded-lg transition-all"
                >
                  Cancel
                </button>
              </div>
            </form>
          )}

          {/* Camera List */}
          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-slate-400 mt-4">Loading cameras...</p>
            </div>
          ) : cameras.length === 0 ? (
            <div className="text-center py-12">
              <svg className="w-16 h-16 mx-auto text-slate-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p className="text-slate-400">No cameras configured yet</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {cameras.map((camera) => (
                <div
                  key={camera.id}
                  className="bg-slate-800/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="text-white font-bold text-lg">{camera.name}</h3>
                        <div className={`w-2 h-2 rounded-full ${getStatusColor(camera.status)}`}></div>
                      </div>
                      <p className="text-slate-400 text-sm">{camera.location}</p>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleEdit(camera)}
                        className="text-cyan-400 hover:text-cyan-300 transition-colors"
                        title="Edit"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                      </button>
                      <button
                        onClick={() => handleDelete(camera.id)}
                        className="text-red-400 hover:text-red-300 transition-colors"
                        title="Delete"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-sm mb-3">
                    <div>
                      <span className="text-slate-500">Type:</span>
                      <span className="text-white ml-2">
                        {camera.type === "webcam" ? "💻 Webcam" : "📹 IP Camera"}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500">Resolution:</span>
                      <span className="text-white ml-2">
                        {camera.resolution_width}x{camera.resolution_height}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500">FPS:</span>
                      <span className="text-white ml-2">{camera.fps}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">Status:</span>
                      <span className={`ml-2 ${
                        camera.status === "online" ? "text-emerald-400" :
                        camera.status === "error" ? "text-red-400" :
                        camera.status === "connecting" ? "text-yellow-400" :
                        "text-slate-400"
                      }`}>
                        {camera.status}
                      </span>
                    </div>
                  </div>

                  {camera.type === "ip_camera" && (
                    <div className="mb-3">
                      <span className="text-slate-500 text-sm">RTSP URL:</span>
                      <p className="text-slate-300 text-xs font-mono mt-1 truncate">
                        {camera.rtsp_url}
                      </p>
                    </div>
                  )}

                  {camera.type === "ip_camera" && (
                    <button
                      onClick={() => testConnection(camera.id)}
                      disabled={loading}
                      className="w-full bg-slate-700 hover:bg-slate-600 text-white text-sm font-semibold py-2 px-4 rounded-lg transition-all disabled:opacity-50"
                    >
                      Test Connection
                    </button>
                  )}

                  {camera.error_message && (
                    <div className="mt-2 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
                      <p className="text-red-400 text-xs">
                        ⚠️ {camera.error_message}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
