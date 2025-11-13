// src/components/FeedbackWidget.jsx
import React, { useState, useEffect } from "react";

export default function FeedbackWidget({ detection }) {
  const [open, setOpen] = useState(false);
  const [correction, setCorrection] = useState("");
  const [notes, setNotes] = useState("");

  // Defensive: if detection undefined, show placeholder until prop available
  useEffect(() => {
    if (!detection) {
      console.warn("FeedbackWidget: detection prop is undefined");
    }
  }, [detection]);

  const submit = async (type, label) => {
    // Basic client-side guard
    if (!detection?.id) {
      alert("No detection selected to attach feedback to.");
      return;
    }

    // Example POST - adapt base URL as needed
    try {
      // If you want to actually call backend, uncomment and adjust:
      // await axios.post(`/api/detections/${detection.id}/feedback`, {
      //   user_id: "student1",
      //   feedback_label: label,
      //   feedback_type: type,
      //   notes,
      // });
      console.log("Feedback submit (test):", {
        detection_id: detection.id,
        type,
        label,
        notes,
      });
      alert("Feedback submitted (test)");
      setOpen(false);
    } catch (err) {
      console.error("Feedback submit error:", err);
      alert("Failed to submit feedback");
    }
  };

//   If detection not provided, render a small placeholder
  if (!detection) {
    return (
      <div style={{
        border: "1px dashed #374151",
        padding: 12,
        borderRadius: 8,
        background: "#071025",
        color: "#9CA3AF",
        maxWidth: 360,
        margin: 12
      }}>
        <div style={{fontSize: 14}}>
          Waiting for detection data...
        </div>
      </div>
    );
  }

  // Now safe to access detection fields using optional chaining
  const label = detection?.detection_label ?? "Unknown";
  const conf = typeof detection?.detection_confidence === "number"
    ? `${(detection.detection_confidence * 100).toFixed(0)}%`
    : "";

  return (
    <div style={{
      border: "1px solid #2b2b2b",
      padding: 12,
      borderRadius: 8,
      background: "#0b1220",
      color: "white",
      maxWidth: 360,
      margin: 12
    }}>
      <div style={{fontSize: 14, marginBottom: 6}}>
        <strong>Prediction:</strong>{" "}
        {label}{" "}
        <small style={{color: "#93c5fd"}}>{conf}</small>
      </div>

      <div style={{display: "flex", gap: 8}}>
        <button
          onClick={() => submit("confirm", label)}
          className="px-3 py-1 rounded"
          style={{background: "#16a34a", color: "white"}}
        >
          👍 Confirm
        </button>
        <button
          onClick={() => setOpen(true)}
          className="px-3 py-1 rounded"
          style={{background: "#dc2626", color: "white"}}
        >
          👎 Correct
        </button>
      </div>

      {open && (
        <div style={{marginTop: 8}}>
          <div style={{marginBottom: 6}}>
            <input
              value={correction}
              onChange={(e) => setCorrection(e.target.value)}
              placeholder="Correct label (e.g., Fighting)"
              style={{width: "100%", padding: 8, borderRadius: 6}}
            />
          </div>
          <div style={{marginBottom: 6}}>
            <input
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Notes (optional)"
              style={{width: "100%", padding: 8, borderRadius: 6}}
            />
          </div>
          <div style={{display: "flex", gap: 8}}>
            <button
              onClick={() => submit("correct", correction || label)}
              className="px-3 py-1 rounded"
              style={{background: "#2563eb", color: "white"}}
            >
              Submit
            </button>
            <button
              onClick={() => setOpen(false)}
              className="px-3 py-1 rounded"
              style={{background: "#6b7280", color: "white"}}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
