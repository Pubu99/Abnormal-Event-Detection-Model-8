import React, { useEffect, useState } from "react";
import axios from "axios";

function AdminPanel({ onClose }) {
  const [scheduleObj, setScheduleObj] = useState(null);
  const [editedSchedule, setEditedSchedule] = useState("");
  const [scheduleErrors, setScheduleErrors] = useState({});
  const [isFormValid, setIsFormValid] = useState(true);
  const [logsDate, setLogsDate] = useState("");
  const [logs, setLogs] = useState([]);
  const [statusMsg, setStatusMsg] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    fetchSchedule();
  }, []);

  async function fetchSchedule() {
    try {
      const res = await axios.get("/api/rl/schedule");
      // API returns { success: True, schedule: { ... } }
      const sched = (res && res.data && res.data.schedule) ? res.data.schedule : res.data;
      setScheduleObj(sched || {});
      setEditedSchedule(JSON.stringify(sched || {}, null, 2));
    } catch (err) {
      // Silently handle 404 - RL endpoints may not be available
      console.warn("RL schedule endpoint not available:", err.message);
      setScheduleObj({});
      setEditedSchedule(JSON.stringify({}, null, 2));
    }
  }

  // Helper to format hour int -> HH:MM for time input
  function hourToTimeStr(h) {
    try {
      const hh = Number(h) || 0;
      return (hh < 10 ? '0' + hh : '' + hh) + ':00';
    } catch (e) { return '00:00'; }
  }

  function daysToNames(daysArr) {
    try {
      if (!Array.isArray(daysArr) || daysArr.length === 0) return 'None';
      const names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
      if (daysArr.length === 7) return 'Every day';
      return daysArr.map(i => names[i] || '?').join(', ');
    } catch (e) { return 'Unknown'; }
  }

  // validate scheduleObj ranges and set errors
  function validateSchedule(s) {
    const errs = {};
    if (!s) return { valid: false, errors: { schedule: 'No schedule loaded' } };
    const min_new = Number(s.min_new || 0);
    if (!(Number.isInteger(min_new) && min_new >= 1)) errs.min_new = 'Min new must be an integer >= 1';
    const sh = Number(s.start_hour || 0);
    const eh = Number(s.end_hour || 0);
    if (!(Number.isInteger(sh) && sh >= 0 && sh <= 23)) errs.start_hour = 'Start hour must be 0-23';
    if (!(Number.isInteger(eh) && eh >= 0 && eh <= 23)) errs.end_hour = 'End hour must be 0-23';
    const cpu = Number(s.cpu_threshold_percent == null ? NaN : s.cpu_threshold_percent);
    if (!(Number.isFinite(cpu) && cpu >= 0 && cpu <= 100)) errs.cpu_threshold_percent = 'CPU threshold must be 0-100';
    const gutil = Number(s.gpu_util_threshold_percent == null ? NaN : s.gpu_util_threshold_percent);
    if (!(Number.isFinite(gutil) && gutil >= 0 && gutil <= 100)) errs.gpu_util_threshold_percent = 'GPU util must be 0-100';
    const gmem = Number(s.gpu_mem_threshold_percent == null ? NaN : s.gpu_mem_threshold_percent);
    if (!(Number.isFinite(gmem) && gmem >= 0 && gmem <= 100)) errs.gpu_mem_threshold_percent = 'GPU mem must be 0-100';
    const cd = Number(s.cooldown_seconds == null ? NaN : s.cooldown_seconds);
    if (!(Number.isFinite(cd) && cd >= 0)) errs.cooldown_seconds = 'Cooldown must be >= 0';
    // days: if provided, ensure ints 0-6
    if (s.days && Array.isArray(s.days)) {
      const bad = s.days.some(d => !(Number.isInteger(d) && d >= 0 && d <= 6));
      if (bad) errs.days = 'Days must be integers 0 (Mon) .. 6 (Sun)';
    }
    const valid = Object.keys(errs).length === 0;
    return { valid, errors: errs };
  }

  // Re-validate whenever scheduleObj changes
  useEffect(() => {
    const { valid, errors } = validateSchedule(scheduleObj || {});
    setScheduleErrors(errors);
    setIsFormValid(valid);
  }, [scheduleObj]);

  // Helper to get hour int from time input like "22:00"
  function timeStrToHour(t) {
    try {
      if (!t) return 0;
      const parts = t.split(':');
      return parseInt(parts[0], 10) || 0;
    } catch (e) { return 0; }
  }

  async function saveSchedule() {
    setIsSaving(true);
    setStatusMsg("");
    try {
      // Build payload from scheduleObj (form state) if available, otherwise fall back to editedSchedule JSON
      let payload = null;
      if (scheduleObj) {
        payload = Object.assign({}, scheduleObj);
      }

      if (!payload) {
        try {
          payload = JSON.parse(editedSchedule);
        } catch (pe) {
          setStatusMsg("Cannot save: invalid JSON: " + (pe.message || String(pe)));
          return;
        }
      }

      // Ensure critical numeric fields are numbers
      const norm = (v, fallback) => (v === undefined || v === null || v === '') ? fallback : Number(v);
      payload.min_new = norm(payload.min_new, 50);
      payload.start_hour = norm(payload.start_hour, 0);
      payload.end_hour = norm(payload.end_hour, 6);
      payload.cpu_threshold_percent = norm(payload.cpu_threshold_percent, 80);
      payload.gpu_util_threshold_percent = norm(payload.gpu_util_threshold_percent, 80);
      payload.gpu_mem_threshold_percent = norm(payload.gpu_mem_threshold_percent, 90);
      payload.cooldown_seconds = norm(payload.cooldown_seconds, 300);

      await axios.post("/api/rl/schedule", payload);
      setStatusMsg("Schedule saved");
      fetchSchedule();
    } catch (err) {
      setStatusMsg("Failed to save schedule: " + (err.message || err));
    } finally {
      setIsSaving(false);
    }
  }

  async function triggerManualTrain() {
    if (!window.confirm("Trigger manual retrain now?")) return;
    setStatusMsg("Starting manual retrain...");
    try {
      const res = await axios.post("/api/rl/train");
      setStatusMsg("Retrain requested: " + (res.data.message || "ok"));
    } catch (err) {
      setStatusMsg("Failed to start retrain: " + (err.message || err));
    }
  }

  async function pauseTraining() {
    if (!window.confirm("Pause the running training process?")) return;
    try {
      await axios.post("/api/rl/pause");
      setStatusMsg("Training paused");
    } catch (err) {
      setStatusMsg("Failed to pause: " + (err.message || err));
    }
  }

  async function resumeTraining() {
    try {
      await axios.post("/api/rl/resume");
      setStatusMsg("Training resumed");
    } catch (err) {
      setStatusMsg("Failed to resume: " + (err.message || err));
    }
  }

  async function fetchLogs() {
    if (!logsDate) {
      setStatusMsg("Choose a date to fetch logs");
      return;
    }
    setStatusMsg("Loading logs...");
    try {
      const res = await axios.get(`/api/rl/logs?date=${encodeURIComponent(logsDate)}`);
      setLogs(res.data || []);
      setStatusMsg("Logs loaded: " + (res.data ? res.data.length : 0));
    } catch (err) {
      // Silently handle 404 - RL endpoints may not be available
      console.warn("RL logs endpoint not available:", err.message);
      setLogs([]);
      setStatusMsg("RL logs not available");
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center p-6">
      <div className="w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-lg shadow-2xl overflow-auto" style={{maxHeight: '90vh'}}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <h2 className="text-lg font-semibold text-white">Admin Panel — RL Retrain</h2>
          <div className="flex items-center gap-3">
            <button className="btn-sm text-sm text-slate-300" onClick={onClose}>Close</button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          <section>
            <h3 className="text-sm font-medium text-slate-200">Current Settings</h3>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-slate-300">
              {!scheduleObj ? (
                <div className="text-slate-500">Loading current schedule...</div>
              ) : (
                <>
                  <div><strong>Enabled:</strong> {scheduleObj.enabled ? 'Yes' : 'No'}</div>
                  <div><strong>Min new:</strong> {scheduleObj.min_new ?? '50'}</div>
                  <div><strong>Days:</strong> {daysToNames(scheduleObj.days)}</div>
                  <div><strong>Window (UTC):</strong> {hourToTimeStr(scheduleObj.start_hour)} — {hourToTimeStr(scheduleObj.end_hour)}</div>
                  <div><strong>CPU thresh %:</strong> {scheduleObj.cpu_threshold_percent ?? '80'}</div>
                  <div><strong>GPU util %:</strong> {scheduleObj.gpu_util_threshold_percent ?? '80'}</div>
                  <div><strong>GPU mem %:</strong> {scheduleObj.gpu_mem_threshold_percent ?? '90'}</div>
                  <div><strong>Cooldown s:</strong> {scheduleObj.cooldown_seconds ?? '300'}</div>
                </>
              )}
            </div>
          </section>
          <section>
            <h3 className="text-sm font-medium text-slate-200">Retrain Schedule</h3>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={!!(scheduleObj && scheduleObj.enabled)} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { enabled: e.target.checked }))} />
                  <span className="text-sm text-slate-200">Auto-Retrain Enabled</span>
                </label>
              </div>

              <div>
                <label className="text-sm text-slate-200">Min new experiences</label>
                <input title="Minimum number of new RL experiences required to trigger an auto-retrain" type="number" min={1} value={(scheduleObj && scheduleObj.min_new) || ''} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { min_new: Number(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.min_new && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.min_new}</div>}
              </div>

              <div className="md:col-span-2">
                <label className="text-sm text-slate-200">Days to run (toggle)</label>
                <div className="flex flex-wrap gap-2 mt-2">
                  {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].map((d, idx) => {
                    const active = scheduleObj && Array.isArray(scheduleObj.days) && scheduleObj.days.indexOf(idx) !== -1;
                    return (
                      <button key={d} onClick={() => {
                        const days = new Set((scheduleObj && scheduleObj.days) || []);
                        if (days.has(idx)) days.delete(idx); else days.add(idx);
                        setScheduleObj(Object.assign({}, scheduleObj || {}, { days: Array.from(days).sort() }));
                      }} className={`px-3 py-1 rounded ${active ? 'bg-cyan-600 text-white' : 'bg-slate-800 text-slate-300'}`}>
                        {d}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="text-sm text-slate-200">Start time (UTC)</label>
                <input title="Hour when the retrain window starts (UTC)" type="time" value={scheduleObj ? hourToTimeStr(scheduleObj.start_hour) : '00:00'} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { start_hour: timeStrToHour(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.start_hour && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.start_hour}</div>}
              </div>

              <div>
                <label className="text-sm text-slate-200">End time (UTC)</label>
                <input title="Hour when the retrain window ends (exclusive) (UTC)" type="time" value={scheduleObj ? hourToTimeStr(scheduleObj.end_hour) : '06:00'} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { end_hour: timeStrToHour(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.end_hour && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.end_hour}</div>}
              </div>

              <div>
                <label className="text-sm text-slate-200">CPU threshold %</label>
                <input title="Pause training if system CPU percentage is >= this value" type="number" min={0} max={100} value={(scheduleObj && scheduleObj.cpu_threshold_percent) || ''} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { cpu_threshold_percent: Number(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.cpu_threshold_percent && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.cpu_threshold_percent}</div>}
              </div>

              <div>
                <label className="text-sm text-slate-200">GPU util threshold %</label>
                <input title="Pause training if GPU utilization% is >= this value (if GPUs detected)" type="number" min={0} max={100} value={(scheduleObj && scheduleObj.gpu_util_threshold_percent) || ''} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { gpu_util_threshold_percent: Number(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.gpu_util_threshold_percent && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.gpu_util_threshold_percent}</div>}
              </div>

              <div>
                <label className="text-sm text-slate-200">GPU mem threshold %</label>
                <input title="Pause training if GPU memory % used is >= this value (if GPUs detected)" type="number" min={0} max={100} value={(scheduleObj && scheduleObj.gpu_mem_threshold_percent) || ''} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { gpu_mem_threshold_percent: Number(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.gpu_mem_threshold_percent && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.gpu_mem_threshold_percent}</div>}
              </div>

              <div>
                <label className="text-sm text-slate-200">Cooldown seconds</label>
                <input title="Seconds to wait after pausing before re-checking resources" type="number" min={0} value={(scheduleObj && scheduleObj.cooldown_seconds) || ''} onChange={(e) => setScheduleObj(Object.assign({}, scheduleObj || {}, { cooldown_seconds: Number(e.target.value) }))} className="w-full mt-1 p-2 bg-slate-800 rounded text-slate-200" />
                {scheduleErrors.cooldown_seconds && <div className="text-xs text-amber-300 mt-1">{scheduleErrors.cooldown_seconds}</div>}
              </div>
            </div>

            <div className="flex items-center gap-3 mt-4">
              <button className="px-3 py-2 bg-cyan-600 rounded text-white" onClick={saveSchedule} disabled={isSaving || !isFormValid} title={isFormValid ? 'Save schedule to backend' : 'Fix form errors before saving'}>
                {isSaving ? 'Saving...' : 'Save Schedule'}
              </button>
              <button className="px-3 py-2 bg-slate-700 rounded text-white" onClick={fetchSchedule}>Reload</button>
              <button className="px-3 py-2 bg-slate-600 rounded text-white" onClick={() => setEditedSchedule(JSON.stringify(scheduleObj || {}, null, 2))}>Show JSON</button>
            </div>
            {!isFormValid && (
              <div className="text-sm text-amber-300 mt-2">Please fix the highlighted fields above before saving.</div>
            )}
          </section>

          <section>
            <h3 className="text-sm font-medium text-slate-200">Manual Controls</h3>
            <div className="flex items-center gap-3 mt-2">
              <button className="px-3 py-2 bg-emerald-600 rounded text-white" onClick={triggerManualTrain}>Trigger Manual Retrain</button>
              <button className="px-3 py-2 bg-yellow-600 rounded text-white" onClick={pauseTraining}>Pause Training</button>
              <button className="px-3 py-2 bg-green-600 rounded text-white" onClick={resumeTraining}>Resume Training</button>
            </div>
          </section>

          <section>
            <h3 className="text-sm font-medium text-slate-200">Retrain Logs</h3>
            <div className="flex items-center gap-3 mt-2">
              <input type="date" value={logsDate} onChange={(e) => setLogsDate(e.target.value)} className="p-2 bg-slate-800 rounded text-slate-200" />
              <button className="px-3 py-2 bg-slate-600 rounded text-white" onClick={fetchLogs}>Fetch Logs</button>
            </div>
            <div className="mt-3 bg-slate-800 rounded p-3 max-h-48 overflow-auto text-xs text-slate-300">
              {logs.length === 0 ? <div className="text-slate-500">No logs to show</div> : (
                logs.map((l, idx) => (
                  <pre key={idx} className="mb-2 whitespace-pre-wrap">{JSON.stringify(l, null, 2)}</pre>
                ))
              )}
            </div>
          </section>

          {statusMsg && <div className="text-sm text-slate-300">{statusMsg}</div>}
        </div>
      </div>
    </div>
  );
}

export default AdminPanel;
