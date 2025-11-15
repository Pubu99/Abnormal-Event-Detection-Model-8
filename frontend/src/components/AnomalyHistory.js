import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';

export default function AnomalyHistory({ onClose }) {
  const [date, setDate] = useState('');
  const [limit, setLimit] = useState(100);
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  const fetchEntries = useCallback(async () => {
    setLoading(true);
    setStatus('');
    try {
      const params = {};
      if (date) params.date = date;
      if (limit) params.limit = limit;
      const res = await axios.get('/api/anomaly-history', { params });
      const data = res && res.data ? res.data : {};
      setEntries(data.entries || []);
      setStatus(`Loaded ${data.count || 0} entries`);
    } catch (err) {
      setStatus('Failed to load: ' + (err.message || err));
    } finally {
      setLoading(false);
    }
  }, [date, limit]);

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  const formatTimestamp = (ts) => {
    try {
      const d = new Date(ts);
      return d.toLocaleString();
    } catch (e) { return ts; }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center p-6">
      <div className="w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-lg shadow-2xl overflow-auto" style={{maxHeight: '90vh'}}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <h2 className="text-lg font-semibold text-white">Anomaly History</h2>
          <div className="flex items-center gap-3">
            <button className="btn-sm text-sm text-slate-300" onClick={onClose}>Close</button>
          </div>
        </div>

        <div className="p-6 space-y-4">
          <div className="flex items-center gap-3">
            <label className="text-sm text-slate-300">Date (UTC):</label>
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className="p-2 bg-slate-800 rounded text-slate-200" />
            <label className="text-sm text-slate-300">Limit:</label>
            <input type="number" value={limit} onChange={(e) => setLimit(Number(e.target.value))} className="w-20 p-2 bg-slate-800 rounded text-slate-200" />
            <button className="px-3 py-2 bg-cyan-600 rounded text-white" onClick={fetchEntries} disabled={loading}>{loading ? 'Loading...' : 'Fetch'}</button>
            <div className="text-sm text-slate-400">{status}</div>
          </div>

          <div className="bg-slate-800 rounded p-3 max-h-72 overflow-auto text-xs text-slate-300">
            {entries.length === 0 ? (
              <div className="text-slate-500">No confirmed anomalies found.</div>
            ) : (
              entries.map((e, idx) => (
                <div key={idx} className="border-b border-slate-700 pb-2 mb-2">
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-white truncate">{e.anomaly_type || e.anomaly_type}</div>
                    <div className="text-slate-400 text-xs">{formatTimestamp(e.recorded_at_utc || e.timestamp || e.recorded_at)}</div>
                  </div>
                  <div className="text-xs text-slate-300 mt-1">
                    <strong>ID:</strong> {e.detection_id || 'n/a'} • <strong>Severity:</strong> {e.severity || 'n/a'} • <strong>Fusion:</strong> {Number(e.fusion_score).toFixed(3)} • <strong>Conf:</strong> {Number(e.confidence).toFixed(3)}
                  </div>
                  <div className="text-xs text-slate-300 mt-2">
                    <strong>Objects:</strong> {(e.detected_objects || []).slice(0,5).map(o=> (typeof o === 'string' ? o : (o.class || JSON.stringify(o)))).join(', ')}
                  </div>
                  {e.user && <div className="text-xs text-slate-400 mt-1"><strong>Confirmed by:</strong> {e.user} {e.comment ? '• ' + e.comment : ''}</div>}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
