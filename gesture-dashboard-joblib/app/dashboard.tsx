'use client';

import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_GESTURE_API_URL ?? 'http://127.0.0.1:8000';
const WS_URL = API_URL.replace(/^http/, 'ws');
const DEFAULT_CLASSES = [
  'call', 'rock', 'like', 'ok', 'one', 'one_down', 'one_left', 'one_right',
  'palm', 'peace', 'dorsal_hand', 'fist', 'zoom_in', 'zoom_out', 'no_gesture',
];
const DEFAULT_ACTIONS: Record<string, string> = {
  call: 'Detect My Face',
  rock: 'Pause Video',
  like: 'Stop / Continue',
  ok: 'Start Recording',
  one: 'Move Up',
  one_down: 'Move Down',
  one_left: 'Move Left',
  one_right: 'Move Right',
  palm: 'Open Palm',
  peace: 'End Recording',
  dorsal_hand: 'Return to Main Position',
  fist: 'Fist (Follow Object phase)',
  zoom_in: 'Zoom In',
  zoom_out: 'Zoom Out',
  no_gesture: 'No Gesture',
};
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

type Artifact = {
  label: string;
  pattern: string;
  present: boolean;
  required: boolean;
  files: string[];
  bytes: number;
};
type Timing = {
  mediapipe_ms?: number;
  feature_ms?: number;
  classifier_ms?: number;
  diagnostics_ms?: number;
  total_ms?: number;
  uncapped_fps?: number;
  configured_fps?: number;
  effective_application_fps?: number;
  frame_budget_ms?: number;
  budget_used_percent?: number;
  headroom_ms?: number;
  ten_fps_capacity_pass?: boolean;
  verdict?: 'PASS' | 'FAIL';
};
type Diagnostic = {
  used_for_runtime: boolean;
  raw_prediction: string;
  prediction: string;
  confidence: number;
  stable_frames: number;
  execute: boolean;
  reason: string;
  classifier_ms: number;
};
type FollowState = {
  state: string;
  step_index: number;
  completed: boolean;
  active: boolean;
  message: string;
  expected_gesture?: string;
  hold_progress?: number;
  confidence?: number;
  source?: string;
};
type Health = {
  status: string;
  operational_ready?: boolean;
  engine: {
    ready: boolean;
    model: {
      ready: boolean;
      available_models: string[];
      selectable_models?: string[];
      selected_model_name: string;
      follow_object_ensemble_ready?: boolean;
      errors: string[];
    };
    mediapipe: { ready: boolean; error?: string };
  };
  config: {
    class_names: string[];
    gesture_to_action: Record<string, string>;
    target_fps: number;
    frame_interval_ms: number;
    frame_budget_ms: number;
    ema_alpha: number;
    confidence_floor: number;
    stable_frames_required: number;
    roi_size_ratio: number;
    follow_hold_seconds: number;
    follow_success_display_seconds: number;
  };
  artifacts: Artifact[];
  online_learning?: {
    ready: boolean;
    accepted_updates: number;
    forced_updates: number;
    rejected_updates: number;
    validation_macro_f1?: number;
    last_updated_utc?: string;
    force_learning_enabled?: boolean;
    backup_count?: number;
    checkpoint_policy: string;
  };
  artifact_integrity?: {
    verified: boolean;
    status: string;
    release_fingerprint?: string;
    checked_files: number;
    errors: string[];
  };
  production?: {
    environment: string;
    production_mode: boolean;
    allow_force_learning: boolean;
    allow_artifact_reload: boolean;
    max_active_sessions: number;
    release_qualification?: {
      status: string;
      current?: boolean;
      pc_release_ready: boolean;
      selected_model?: string;
      gated_macro_f1?: number;
      gated_minimum_per_class_f1?: number;
      deferred_classes?: string[];
      failing_gated_classes?: string[];
      runtime_format?: string;
      message?: string;
    };
  };
  runtime_metrics?: {
    frames_total: number;
    errors_total: number;
    error_rate: number;
    ten_fps_budget_pass_rate: number;
    active_sessions: number;
    peak_sessions: number;
    timing: Record<string, { count: number; mean?: number; p50?: number; p95?: number; p99?: number }>;
  };
};
type Prediction = {
  status: string;
  message?: string;
  model?: string;
  raw_prediction?: string;
  runtime_prediction?: string;
  mapped_action?: string;
  runtime_action?: string;
  action_reason?: string;
  confidence?: number;
  stable_frames?: number;
  held_seconds?: number;
  probabilities?: Record<string, number>;
  raw_probabilities?: Record<string, number>;
  feature_vector?: number[];
  landmarks?: number[][];
  actual_fps?: number;
  timing?: Timing;
  diagnostics?: Record<string, Diagnostic>;
  follow_object?: FollowState;
};
type MetricResponse = { files: string[]; rows: Record<string, Record<string, string>[]> };
type ActionEvent = { action: string; prediction: string; confidence: number; session_id?: string };
type LearningMode = 'audit' | 'safe' | 'force';

function humanize(value?: string) {
  if (!value) return 'Waiting';
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}
function percent(value?: number) { return `${((value ?? 0) * 100).toFixed(1)}%`; }
function milliseconds(value?: number) { return value == null ? '—' : `${value.toFixed(1)} ms`; }
function fps(value?: number | null) { return value == null || !Number.isFinite(value) ? '—' : value.toFixed(2); }

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'live' | 'analytics' | 'feedback' | 'setup'>('live');
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<MetricResponse>({ files: [], rows: {} });
  const [actions, setActions] = useState<ActionEvent[]>([]);
  const [prediction, setPrediction] = useState<Prediction>({ status: 'idle' });
  const [connection, setConnection] = useState<'offline' | 'connecting' | 'online'>('offline');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraFps, setCameraFps] = useState<number | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [followActive, setFollowActive] = useState(false);
  const [successVisible, setSuccessVisible] = useState(false);
  const [feedbackCount, setFeedbackCount] = useState(0);
  const [actualLabel, setActualLabel] = useState('no_gesture');
  const [learningMode, setLearningMode] = useState<LearningMode>('audit');
  const [feedbackNote, setFeedbackNote] = useState('');
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [uploading, setUploading] = useState(false);
  const [snapshotReady, setSnapshotReady] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const landmarkCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const captureTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const successTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef(false);
  const lastSentAtRef = useRef(0);
  const lastSnapshotRef = useRef<string | null>(null);
  const videoFrameTimesRef = useRef<number[]>([]);
  const videoFrameCallbackRef = useRef<number | null>(null);
  const captureAndSendRef = useRef<() => void>(() => undefined);
  const stopCameraRef = useRef<(preservePrediction?: boolean) => void>(() => undefined);

  const targetFps = health?.config.target_fps ?? 10;
  const frameIntervalMs = health?.config.frame_interval_ms ?? 100;
  const stableRequired = health?.config.stable_frames_required ?? 3;
  const classNames = health?.config.class_names ?? DEFAULT_CLASSES;
  const actionMap = health?.config.gesture_to_action ?? DEFAULT_ACTIONS;
  const selectableModels = health?.engine.model.selectable_models ?? health?.engine.model.available_models ?? [];
  const serverReady = Boolean(health?.operational_ready ?? health?.engine.ready);
  const onlineReady = Boolean(health?.online_learning?.ready);
  const forceLearningEnabled = Boolean(health?.production?.allow_force_learning);
  const qualification = health?.production?.release_qualification;
  const runtimeSummary = health?.runtime_metrics;
  const mappedLabel = prediction.mapped_action
    ?? (prediction.runtime_prediction ? actionMap[prediction.runtime_prediction] : undefined)
    ?? 'Waiting for prediction';
  const timing = prediction.timing;
  const pipelineVerdict = timing?.verdict ?? 'WAITING';

  const refreshServerData = async () => {
    try {
      const [healthResponse, metricResponse, actionResponse, feedbackResponse] = await Promise.all([
        fetch(`${API_URL}/api/health`),
        fetch(`${API_URL}/api/metrics`),
        fetch(`${API_URL}/api/actions`),
        fetch(`${API_URL}/api/feedback`),
      ]);
      if (!healthResponse.ok) throw new Error('Health request failed.');
      const healthData = await healthResponse.json() as Health;
      setHealth(healthData);
      const choices = healthData.engine.model.selectable_models ?? healthData.engine.model.available_models;
      setSelectedModel((current) => current && choices.includes(current)
        ? current
        : healthData.engine.model.selected_model_name || choices[0] || '');
      if (metricResponse.ok) setMetrics(await metricResponse.json() as MetricResponse);
      if (actionResponse.ok) {
        const actionData = await actionResponse.json() as { recent?: ActionEvent[] };
        setActions(actionData.recent ?? []);
      }
      if (feedbackResponse.ok) {
        const feedbackData = await feedbackResponse.json() as { count?: number };
        setFeedbackCount(feedbackData.count ?? 0);
      }
    } catch {
      setHealth(null);
    }
  };

  useEffect(() => {
    const initial = setTimeout(refreshServerData, 0);
    const refresh = setInterval(refreshServerData, 10000);
    return () => { clearTimeout(initial); clearInterval(refresh); };
  }, []);

  useEffect(() => () => {
    if (captureTimerRef.current) clearTimeout(captureTimerRef.current);
    if (successTimerRef.current) clearTimeout(successTimerRef.current);
    socketRef.current?.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  const probabilityRows = useMemo(() => {
    const values = prediction.probabilities ?? Object.fromEntries(classNames.map((name) => [name, 0]));
    return classNames
      .map((name) => ({ name, value: values[name] ?? 0 }))
      .sort((first, second) => second.value - first.value);
  }, [prediction.probabilities, classNames]);

  const diagnostics = useMemo(
    () => Object.entries(prediction.diagnostics ?? {}),
    [prediction.diagnostics],
  );

  const drawLandmarks = (landmarks?: number[][]) => {
    const canvas = landmarkCanvasRef.current;
    if (!canvas) return;
    const size = Math.max(1, Math.round(canvas.clientWidth || 400));
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, size, size);
    if (!landmarks || landmarks.length !== 21) return;
    context.lineWidth = Math.max(2, size / 180);
    context.strokeStyle = 'rgba(73, 217, 209, .92)';
    context.beginPath();
    HAND_CONNECTIONS.forEach(([from, to]) => {
      context.moveTo(landmarks[from][0] * size, landmarks[from][1] * size);
      context.lineTo(landmarks[to][0] * size, landmarks[to][1] * size);
    });
    context.stroke();
    landmarks.forEach(([x, y], index) => {
      context.beginPath();
      context.fillStyle = index === 0 ? '#ffb454' : '#f4f7ff';
      context.arc(x * size, y * size, Math.max(2.4, size / 120), 0, Math.PI * 2);
      context.fill();
    });
  };

  useEffect(() => { drawLandmarks(prediction.landmarks); }, [prediction.landmarks]);

  const scheduleCapture = (delayMs = 0) => {
    if (captureTimerRef.current) clearTimeout(captureTimerRef.current);
    captureTimerRef.current = setTimeout(() => captureAndSendRef.current(), Math.max(0, delayMs));
  };

  const captureAndSend = () => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    const socket = socketRef.current;
    if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.readyState < 2) return;
    if (inFlightRef.current || socket.bufferedAmount > 0) return;
    const side = Math.min(video.videoWidth, video.videoHeight) * (health?.config.roi_size_ratio ?? 0.92);
    const sourceX = (video.videoWidth - side) / 2;
    const sourceY = (video.videoHeight - side) / 2;
    canvas.width = 320;
    canvas.height = 320;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, sourceX, sourceY, side, side, 0, 0, canvas.width, canvas.height);
    context.restore();
    lastSnapshotRef.current = canvas.toDataURL('image/jpeg', 0.88);
    setSnapshotReady(true);
    canvas.toBlob((blob) => {
      if (!blob || socket.readyState !== WebSocket.OPEN || inFlightRef.current) {
        scheduleCapture(frameIntervalMs);
        return;
      }
      inFlightRef.current = true;
      lastSentAtRef.current = performance.now();
      socket.send(blob);
    }, 'image/jpeg', 0.88);
  };

  useEffect(() => {
    captureAndSendRef.current = captureAndSend;
  });

  const stopVideoFrameCounter = () => {
    const video = videoRef.current;
    if (video && videoFrameCallbackRef.current != null && typeof video.cancelVideoFrameCallback === 'function') {
      video.cancelVideoFrameCallback(videoFrameCallbackRef.current);
    }
    videoFrameCallbackRef.current = null;
    videoFrameTimesRef.current = [];
  };

  const startVideoFrameCounter = () => {
    const video = videoRef.current;
    if (!video || typeof video.requestVideoFrameCallback !== 'function') return;
    const countFrame = (now: number) => {
      const times = [...videoFrameTimesRef.current, now].filter((value) => now - value <= 2000);
      videoFrameTimesRef.current = times;
      if (times.length > 1) setCameraFps((times.length - 1) * 1000 / (times[times.length - 1] - times[0]));
      videoFrameCallbackRef.current = video.requestVideoFrameCallback(countFrame);
    };
    videoFrameCallbackRef.current = video.requestVideoFrameCallback(countFrame);
  };

  const stopCamera = (preservePrediction = false) => {
    if (captureTimerRef.current) clearTimeout(captureTimerRef.current);
    captureTimerRef.current = null;
    inFlightRef.current = false;
    stopVideoFrameCounter();
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setConnection('offline');
    setFollowActive(false);
    if (!preservePrediction) drawLandmarks();
  };

  useEffect(() => {
    stopCameraRef.current = stopCamera;
  });

  const showSuccess = () => {
    stopCameraRef.current(true);
    setSuccessVisible(true);
    if (successTimerRef.current) clearTimeout(successTimerRef.current);
    successTimerRef.current = setTimeout(
      () => setSuccessVisible(false),
      (health?.config.follow_success_display_seconds ?? 8) * 1000,
    );
  };

  const startCamera = async (beginFollowImmediately = false) => {
    if (cameraActive) return;
    setSuccessVisible(false);
    setConnection('connecting');
    setPrediction({ status: 'starting', message: 'Requesting camera access…' });
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: targetFps, max: targetFps },
            facingMode: 'user',
          },
          audio: false,
        });
      } catch (error) {
        if (!(error instanceof DOMException) || error.name !== 'OverconstrainedError') throw error;
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        });
      }
      streamRef.current = stream;
      const settings = stream.getVideoTracks()[0]?.getSettings();
      setCameraFps(settings?.frameRate ?? null);
      if (!videoRef.current) throw new Error('Video preview is unavailable.');
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      startVideoFrameCounter();
      const socket = new WebSocket(`${WS_URL}/ws/live`);
      socketRef.current = socket;
      socket.onopen = () => {
        setConnection('online');
        setCameraActive(true);
        if (selectedModel) socket.send(JSON.stringify({ type: 'select_model', model: selectedModel }));
        if (beginFollowImmediately) socket.send(JSON.stringify({ type: 'start_follow_object' }));
        scheduleCapture(0);
      };
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'prediction') {
          inFlightRef.current = false;
          setPrediction(message);
          setFollowActive(Boolean(message.follow_object?.active));
          if (message.runtime_action && message.runtime_action !== 'Wait / No Action') {
            setActions((current) => [{
              action: message.runtime_action,
              prediction: message.runtime_prediction,
              confidence: message.confidence,
            }, ...current].slice(0, 30));
          }
          if (message.follow_object?.completed) {
            showSuccess();
            return;
          }
          const elapsed = performance.now() - lastSentAtRef.current;
          scheduleCapture(Math.max(0, frameIntervalMs - elapsed));
        } else if (message.type === 'follow_object_started' || message.type === 'follow_object_stopped') {
          setFollowActive(Boolean(message.follow_object?.active));
          setPrediction((current) => ({ ...current, follow_object: message.follow_object }));
        } else if (message.type === 'error') {
          setPrediction({ status: 'error', message: message.message });
        }
      };
      socket.onerror = () => setPrediction({ status: 'error', message: 'The local inference server could not be reached.' });
      socket.onclose = () => { setConnection('offline'); setCameraActive(false); };
    } catch (error) {
      stopCameraRef.current();
      setPrediction({
        status: 'camera_error',
        message: error instanceof Error ? error.message : 'Camera access failed.',
      });
    }
  };

  const toggleCamera = () => {
    if (cameraActive) stopCameraRef.current();
    else void startCamera(false);
  };

  const beginFollowObject = () => {
    setSuccessVisible(false);
    if (!cameraActive || socketRef.current?.readyState !== WebSocket.OPEN) {
      void startCamera(true);
      return;
    }
    socketRef.current.send(JSON.stringify({ type: 'start_follow_object' }));
  };

  const stopFollowObject = () => {
    socketRef.current?.send(JSON.stringify({ type: 'stop_follow_object' }));
    setFollowActive(false);
  };

  const resetSession = () => {
    socketRef.current?.send(JSON.stringify({ type: 'reset' }));
    lastSnapshotRef.current = null;
    setSnapshotReady(false);
    setFollowActive(false);
    setPrediction({ status: 'reset', message: 'Temporal EMA and Follow Object state reset.' });
    drawLandmarks();
  };

  const changeModel = (event: ChangeEvent<HTMLSelectElement>) => {
    const model = event.target.value;
    setSelectedModel(model);
    socketRef.current?.send(JSON.stringify({ type: 'select_model', model }));
  };

  const predictUploadedImage = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('image', file);
      const query = selectedModel ? `?model_name=${encodeURIComponent(selectedModel)}` : '';
      const response = await fetch(`${API_URL}/api/predict-image${query}`, { method: 'POST', body: form });
      const result = await response.json() as Prediction;
      setPrediction(result);
      setActiveTab('live');
      lastSnapshotRef.current = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.readAsDataURL(file);
      });
      setSnapshotReady(true);
    } catch {
      setPrediction({ status: 'error', message: 'Uploaded-image prediction failed.' });
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const submitFeedback = async () => {
    setFeedbackMessage(learningMode === 'audit' ? 'Saving reviewed sample…' : 'Validating candidate update…');
    try {
      const response = await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actual_label: actualLabel,
          predicted_label: prediction.runtime_prediction ?? 'no_gesture',
          runtime_action: prediction.runtime_action ?? 'Wait / No Action',
          confidence: prediction.confidence ?? 0,
          model_name: prediction.model ?? selectedModel ?? 'unknown',
          probabilities: prediction.probabilities ?? {},
          feature_vector: prediction.feature_vector ?? null,
          landmarks: prediction.landmarks ?? null,
          note: feedbackNote,
          snapshot_data_url: lastSnapshotRef.current,
          learning_mode: learningMode,
        }),
      });
      const payload = await response.json() as { detail?: string; online_update?: { message?: string } };
      if (!response.ok) throw new Error(payload.detail ?? 'Feedback failed.');
      setFeedbackMessage(payload.online_update?.message ?? 'Feedback saved.');
      setFeedbackNote('');
      setFeedbackCount((count) => count + 1);
      if (learningMode !== 'audit') await refreshServerData();
    } catch (error) {
      setFeedbackMessage(error instanceof Error ? error.message : 'Feedback could not be saved.');
    }
  };

  const reloadArtifacts = async () => {
    try {
      await fetch(`${API_URL}/api/reload`, { method: 'POST' });
      await refreshServerData();
    } catch {
      setFeedbackMessage('The local server is not available.');
    }
  };

  const follow = prediction.follow_object;
  const followStep = follow?.step_index ?? 0;
  const followMessage = follow?.message ?? 'Press Begin Follow Object to start the dedicated sequence.';

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">G</span>
          <div><p className="eyebrow">WINDOWS · LOCAL INFERENCE</p><h1>Gesture Control Lab</h1></div>
        </div>
        <div className="topbar-status">
          <span className={`status-dot ${connection === 'online' ? 'is-online' : ''}`} />
          <span>{connection === 'online' ? 'Live session connected' : health ? 'Local server ready' : 'Server offline'}</span>
          <span className="model-pill">Strict {targetFps} FPS</span>
        </div>
      </header>

      <nav className="nav-rail" aria-label="Dashboard sections">
        {(['live', 'analytics', 'feedback', 'setup'] as const).map((tab, index) => (
          <button className={`nav-item ${activeTab === tab ? 'active' : ''}`} type="button" key={tab} onClick={() => setActiveTab(tab)}>
            <span>0{index + 1}</span>{humanize(tab)}
          </button>
        ))}
        <div className="rail-footer">
          <p>Runtime contract</p>
          <strong>EMA α {health?.config.ema_alpha ?? 0.65}</strong>
          <span>{stableRequired} stable frames · 76-D</span>
        </div>
      </nav>

      <section className="workspace">
        {activeTab === 'live' && <>
          <div className="section-heading">
            <div><p className="eyebrow">REAL-TIME RECOGNITION</p><h2>10 FPS live qualification</h2></div>
            <div className="heading-actions">
              <label className="secondary-button file-button">
                {uploading ? 'Processing…' : 'Test image'}
                <input type="file" accept="image/*" onChange={predictUploadedImage} disabled={uploading || !serverReady} />
              </label>
              <button className="secondary-button" type="button" onClick={resetSession}>Reset</button>
              <button className="primary-button" type="button" onClick={toggleCamera} disabled={!serverReady}>
                {cameraActive ? 'Stop camera' : 'Start 10 FPS test'}
              </button>
            </div>
          </div>

          {!serverReady && <div className="setup-banner">
            <strong>Model setup required.</strong>
            <span>The UI and MediaPipe runtime are ready, but your trained v18_17 artifacts are not in the models folder.</span>
            <button type="button" onClick={() => setActiveTab('setup')}>Open setup</button>
          </div>}

          <article className={`qualification-strip ${pipelineVerdict.toLowerCase()}`}>
            <div><span>10 FPS CAPACITY TEST</span><strong>{pipelineVerdict}</strong></div>
            <p>{timing
              ? timing.ten_fps_capacity_pass
                ? `Pipeline uses ${timing.budget_used_percent?.toFixed(1)}% of the 100 ms frame budget. This PC can sustain the configured 10 FPS workload.`
                : `Pipeline needs ${timing.total_ms?.toFixed(1)} ms per frame, above the 100 ms budget. Reduce workload or use faster hardware.`
              : 'Start the camera and hold a gesture to measure end-to-end processing capacity.'}</p>
          </article>

          <div className="dashboard-grid">
            <article className="camera-card panel">
              <div className="panel-header">
                <div><span className={`live-dot ${cameraActive ? 'is-online' : ''}`} /> CAMERA PREVIEW</div>
                <span>Mirror corrected · central 92% ROI</span>
              </div>
              <div className="camera-stage live-stage">
                <video ref={videoRef} muted playsInline className={`camera-video ${cameraActive ? 'active' : ''}`} />
                {followActive && !successVisible && <div className="camera-instruction">{followMessage}</div>}
                {successVisible && <div className="success-overlay" role="status" aria-live="assertive">
                  <span>✓</span><strong>FOLLOW OBJECT<br />DONE SUCCESSFULLY</strong><p>Camera is OFF</p>
                </div>}
                <div className="guide-box live-guide">
                  <span className="corner top-left" /><span className="corner top-right" />
                  <span className="corner bottom-left" /><span className="corner bottom-right" />
                  <canvas ref={landmarkCanvasRef} className="landmark-canvas" />
                  {!cameraActive && !successVisible && <div className="camera-empty">
                    <span className="hand-orbit" /><strong>Camera is ready</strong>
                    <p>Keep one complete hand and wrist inside the large guide.</p>
                  </div>}
                </div>
                <canvas ref={captureCanvasRef} className="capture-canvas" />
              </div>
              <div className="camera-footer">
                <span>{prediction.status === 'predicted' ? `${prediction.model} runtime` : humanize(prediction.status)}</span>
                <span>Device {fps(cameraFps)} FPS · inference capped at {targetFps.toFixed(0)} FPS</span>
              </div>
            </article>

            <aside className="result-stack">
              <article className="result-card accent-mint">
                <p>RUNTIME PREDICTION</p><strong>{humanize(prediction.runtime_prediction)}</strong>
                <span>{prediction.message ?? `${percent(prediction.confidence)} confidence · ${prediction.stable_frames ?? 0}/${stableRequired} stable`}</span>
              </article>
              <article className="result-card label-card" aria-live="polite">
                <p>LABEL</p><strong>{mappedLabel}</strong>
                <span>{prediction.runtime_prediction
                  ? `Mapped from ${humanize(prediction.runtime_prediction)}`
                  : 'The mapped action will appear after prediction.'}</span>
              </article>
              <article className="result-card accent-orange">
                <p>RUNTIME ACTION</p><strong>{prediction.runtime_action ?? 'Wait / No Action'}</strong>
                <span>{prediction.action_reason ?? 'Confidence and stability gates are active.'}</span>
              </article>
              <article className="sequence-card panel">
                <div className="panel-header"><div>FOLLOW OBJECT</div><span>{followActive ? 'ACTIVE' : 'OPT-IN'}</span></div>
                <div className="sequence-flow">
                  {['Palm', 'Fist', 'Palm'].map((label, index) => <span key={`${label}-${index}`} className={`sequence-node ${followActive && followStep === index ? 'current' : followStep > index ? 'done' : ''}`}>{label}</span>)}
                </div>
                <div className="follow-progress"><i style={{ width: `${Math.round((follow?.hold_progress ?? 0) * 100)}%` }} /></div>
                <p className="sequence-message">{followMessage}</p>
                <button className={followActive ? 'danger-button' : 'follow-button'} type="button" onClick={followActive ? stopFollowObject : beginFollowObject} disabled={!serverReady}>
                  {followActive ? 'Stop procedure' : 'Begin Follow Object'}
                </button>
              </article>
              <div className="model-control">
                <label htmlFor="model-select">Runtime model</label>
                <select id="model-select" value={selectedModel} onChange={changeModel} disabled={!selectableModels.length || followActive}>
                  {selectableModels.map((model) => <option key={model}>{model}</option>)}
                  {!selectableModels.length && <option>No model loaded</option>}
                </select>
              </div>
            </aside>

            <article className="probability-card panel">
              <div className="panel-header"><div>CLASS PROBABILITIES</div><span>EMA · ALL {classNames.length} CLASSES</span></div>
              <div className="probability-list full-list">
                {probabilityRows.map(({ name, value }) => <div className="probability-row" key={name}>
                  <span>{humanize(name)}</span><div><i style={{ width: `${Math.min(100, value * 100)}%` }} /></div><strong>{percent(value)}</strong>
                </div>)}
              </div>
            </article>

            <article className="performance-card panel">
              <div className="panel-header"><div>PIPELINE HEALTH</div><span>{pipelineVerdict}</span></div>
              <div className="metric-grid metric-grid-wide">
                <div><span>Camera FPS</span><strong>{fps(cameraFps)}</strong></div>
                <div><span>Configured limit</span><strong>{targetFps.toFixed(2)}</strong></div>
                <div><span>Effective app FPS</span><strong>{fps(timing?.effective_application_fps ?? prediction.actual_fps)}</strong></div>
                <div><span>Uncapped pipeline</span><strong>{fps(timing?.uncapped_fps)}</strong></div>
                <div><span>Total pipeline</span><strong>{milliseconds(timing?.total_ms)}</strong></div>
                <div><span>Frame budget</span><strong>{milliseconds(timing?.frame_budget_ms ?? 100)}</strong></div>
                <div><span>Budget used</span><strong>{timing?.budget_used_percent == null ? '—' : `${timing.budget_used_percent.toFixed(1)}%`}</strong></div>
                <div><span>Headroom</span><strong>{milliseconds(timing?.headroom_ms)}</strong></div>
              </div>
            </article>
          </div>
        </>}

        {activeTab === 'analytics' && <>
          <div className="section-heading">
            <div><p className="eyebrow">MODEL OBSERVABILITY</p><h2>Diagnostics & evidence</h2></div>
            <button className="secondary-button" type="button" onClick={refreshServerData}>Refresh data</button>
          </div>
          <div className="analytics-grid">
            <article className="panel summary-panel">
              <div className="panel-header"><div>10 FPS SUMMARY</div><span>END-TO-END</span></div>
              <div className="summary-metrics">
                <div><span>Verdict</span><strong className={pipelineVerdict === 'PASS' ? 'text-pass' : 'text-warn'}>{pipelineVerdict}</strong></div>
                <div><span>Runtime model</span><strong>{(prediction.model ?? selectedModel) || '—'}</strong></div>
                <div><span>MediaPipe</span><strong>{milliseconds(timing?.mediapipe_ms)}</strong></div>
                <div><span>Classifier</span><strong>{milliseconds(timing?.classifier_ms)}</strong></div>
              </div>
            </article>

            <article className="panel diagnostics-table-panel">
              <div className="panel-header"><div>MODEL DIAGNOSTICS</div><span>SVM · MLP · ONLINE</span></div>
              {diagnostics.length ? <div className="table-wrap"><table className="diagnostics-table">
                <thead><tr><th>Model</th><th>Runtime</th><th>Raw</th><th>EMA result</th><th>Confidence</th><th>Stable</th><th>Decision</th><th>Time</th></tr></thead>
                <tbody>{diagnostics.map(([name, row]) => <tr key={name}>
                  <td>{name}</td><td>{row.used_for_runtime ? 'YES' : '—'}</td><td>{humanize(row.raw_prediction)}</td><td>{humanize(row.prediction)}</td>
                  <td>{percent(row.confidence)}</td><td>{row.stable_frames}</td><td>{row.execute ? 'EXECUTE' : row.reason}</td><td>{milliseconds(row.classifier_ms)}</td>
                </tr>)}</tbody>
              </table></div> : <p className="empty-note">Start a live session to populate per-model diagnostics.</p>}
            </article>

            <article className="panel action-log">
              <div className="panel-header"><div>ACTION HISTORY</div><span>{actions.length} RECENT</span></div>
              <div className="log-list">{actions.length ? actions.slice(0, 12).map((item, index) => <div key={`${item.action}-${index}`}>
                <span>{String(index + 1).padStart(2, '0')}</span><strong>{item.action}</strong><em>{humanize(item.prediction)} · {percent(item.confidence)}</em>
              </div>) : <p className="empty-note">Stable confirmed actions will appear here.</p>}</div>
            </article>

            <article className="panel metric-files">
              <div className="panel-header"><div>NOTEBOOK EVALUATION EXPORTS</div><span>CSV</span></div>
              {metrics.files.length ? <ul>{metrics.files.map((file) => <li key={file}><span>{file}</span><strong>{metrics.rows[file]?.length ?? 0} rows</strong></li>)}</ul>
                : <p className="empty-note">Copy the notebook evaluation CSV files into the models folder to inspect them here.</p>}
            </article>

            <article className="panel class-map">
              <div className="panel-header"><div>COMMAND TAXONOMY</div><span>{classNames.length} CLASSES</span></div>
              <div className="class-map-grid">{classNames.map((name) => <div key={name}><span>{humanize(name)}</span><strong>{actionMap[name]}</strong></div>)}</div>
            </article>
          </div>
        </>}

        {activeTab === 'feedback' && <>
          <div className="section-heading">
            <div><p className="eyebrow">REVIEWED ONLINE LEARNING</p><h2>Correct, validate, persist</h2></div>
            <span className="count-pill">{feedbackCount} reviewed samples</span>
          </div>
          <div className="feedback-layout">
            <article className="panel feedback-preview">
              <div className="panel-header"><div>LAST PREDICTION</div><span>{prediction.model ?? 'NO MODEL'}</span></div>
              <div className="feedback-prediction"><p>The system predicted</p><strong>{humanize(prediction.runtime_prediction)}</strong><span>{percent(prediction.confidence)} confidence</span></div>
              <div className="feedback-note"><strong>Guarded learning</strong><p>Safe Learn accepts a candidate only when validation macro-F1, per-class F1, and target probability gates pass. Force Learn is disabled by default in production.</p></div>
              <div className="online-status-grid">
                <div><span>Online cache</span><strong>{onlineReady ? 'Ready' : 'Needs Drive files'}</strong></div>
                <div><span>Accepted</span><strong>{health?.online_learning?.accepted_updates ?? 0}</strong></div>
                <div><span>Forced</span><strong>{health?.online_learning?.forced_updates ?? 0}</strong></div>
                <div><span>Rejected</span><strong>{health?.online_learning?.rejected_updates ?? 0}</strong></div>
              </div>
            </article>

            <article className="panel feedback-form">
              <div className="panel-header"><div>CORRECT THE LABEL</div><span>USER CONFIRMATION REQUIRED</span></div>
              <div className="form-body">
                <label>Actual gesture<select value={actualLabel} onChange={(event) => setActualLabel(event.target.value)}>
                  {classNames.map((name) => <option key={name} value={name}>{humanize(name)} — {actionMap[name]}</option>)}
                </select></label>
                <label>Learning action<select value={learningMode} onChange={(event) => setLearningMode(event.target.value as LearningMode)}>
                  <option value="audit">Save feedback only</option>
                  <option value="safe" disabled={!onlineReady}>Safe Learn — validation gated</option>
                  <option value="force" disabled={!onlineReady || !forceLearningEnabled}>Force Learn — administrator only</option>
                </select></label>
                <label>Reviewer note<textarea value={feedbackNote} onChange={(event) => setFeedbackNote(event.target.value)} placeholder="Describe lighting, angle, occlusion, or the confusion you observed." rows={4} /></label>
                <div className="form-details"><span>Prediction: {humanize(prediction.runtime_prediction)}</span><span>Snapshot: {snapshotReady ? 'ready' : 'not captured'}</span><span>76-D features: {prediction.feature_vector?.length === 76 ? 'ready' : 'not available'}</span></div>
                {learningMode === 'force' && <p className="force-warning">Force Learn can reduce other class accuracy and requires an explicitly enabled, authenticated maintenance session.</p>}
                <button className="primary-button wide" type="button" onClick={submitFeedback} disabled={!prediction.runtime_prediction || (learningMode !== 'audit' && prediction.feature_vector?.length !== 76)}>Submit reviewed feedback</button>
                {feedbackMessage && <p className="form-message">{feedbackMessage}</p>}
              </div>
            </article>
          </div>
        </>}

        {activeTab === 'setup' && <>
          <div className="section-heading">
            <div><p className="eyebrow">WINDOWS LOCAL DEPLOYMENT</p><h2>Setup & model artifacts</h2></div>
            <button className="primary-button" type="button" onClick={reloadArtifacts} disabled={!health?.production?.allow_artifact_reload}>Reload artifacts</button>
          </div>
          {!!health?.engine.model.errors.length && <div className="setup-banner error-stack">
            <strong>Artifact validation:</strong>
            <span>{health.engine.model.errors.join(' · ')}</span>
          </div>}
          <div className="setup-layout">
            <article className="panel runtime-contract">
              <div className="panel-header"><div>PRODUCTION RELEASE GATE</div><span>{qualification?.pc_release_ready ? 'PASS' : 'ACTION NEEDED'}</span></div>
              <dl>
                <div><dt>Artifact integrity</dt><dd>{health?.artifact_integrity?.verified ? `Verified (${health.artifact_integrity.checked_files})` : 'Failed / unsigned'}</dd></div>
                <div><dt>Qualification report</dt><dd>{qualification?.current ? qualification.status.toUpperCase() : qualification?.status === 'not_run' ? 'NOT RUN' : 'STALE'}</dd></div>
                <div><dt>Gated macro-F1</dt><dd>{qualification?.gated_macro_f1 == null ? '—' : percent(qualification.gated_macro_f1)}</dd></div>
                <div><dt>Minimum class F1</dt><dd>{qualification?.gated_minimum_per_class_f1 == null ? '—' : percent(qualification.gated_minimum_per_class_f1)}</dd></div>
                <div><dt>Deferred quality work</dt><dd>{qualification?.deferred_classes?.map(humanize).join(', ') || 'None'}</dd></div>
                <div><dt>Other failing classes</dt><dd>{qualification?.failing_gated_classes?.length ? qualification.failing_gated_classes.map(humanize).join(', ') : 'None'}</dd></div>
              </dl>
            </article>

            <article className="panel runtime-contract">
              <div className="panel-header"><div>OPERATIONAL SLO WINDOW</div><span>{runtimeSummary?.frames_total ?? 0} FRAMES</span></div>
              <dl>
                <div><dt>Pipeline p50</dt><dd>{milliseconds(runtimeSummary?.timing?.total_ms?.p50)}</dd></div>
                <div><dt>Pipeline p95</dt><dd>{milliseconds(runtimeSummary?.timing?.total_ms?.p95)}</dd></div>
                <div><dt>Pipeline p99</dt><dd>{milliseconds(runtimeSummary?.timing?.total_ms?.p99)}</dd></div>
                <div><dt>10 FPS budget pass</dt><dd>{percent(runtimeSummary?.ten_fps_budget_pass_rate)}</dd></div>
                <div><dt>Error rate</dt><dd>{percent(runtimeSummary?.error_rate)}</dd></div>
                <div><dt>Sessions active / peak</dt><dd>{runtimeSummary?.active_sessions ?? 0} / {runtimeSummary?.peak_sessions ?? 0}</dd></div>
              </dl>
            </article>

            <article className="panel artifact-panel">
              <div className="panel-header"><div>ARTIFACT CHECKLIST</div><span>{health?.status === 'ready' ? 'READY' : 'ACTION NEEDED'}</span></div>
              <div className="artifact-list">{(health?.artifacts ?? []).map((artifact) => <div key={`${artifact.label}-${artifact.pattern}`}>
                <span className={`artifact-state ${artifact.present ? 'present' : ''}`}>{artifact.present ? '✓' : '!'}</span>
                <div><strong>{artifact.label}</strong><p>{artifact.present ? artifact.files.join(', ') : `Expected ${artifact.pattern}`}</p></div>
                <em>{artifact.present ? `${(artifact.bytes / 1024).toFixed(1)} KB` : artifact.required ? 'Required' : 'Optional path'}</em>
              </div>)}</div>
            </article>

            <article className="panel setup-steps">
              <div className="panel-header"><div>FIRST-TIME SETUP</div><span>4 STEPS</span></div>
              <ol>
                <li><span>01</span><div><strong>Run setup once</strong><p>Double-click setup_dashboard.bat to create Python and frontend environments.</p></div></li>
                <li><span>02</span><div><strong>Copy trusted Drive exports</strong><p>Use scripts\import_models.ps1 with the folder downloaded from your own Google Drive.</p></div></li>
                <li><span>03</span><div><strong>Start the application</strong><p>Double-click start_dashboard.bat. Both local services start and this dashboard opens.</p></div></li>
                <li><span>04</span><div><strong>Run the 10 FPS test</strong><p>Start the camera. PASS means total measured work fits inside the 100 ms frame budget.</p></div></li>
              </ol>
            </article>

            <article className="panel runtime-contract">
              <div className="panel-header"><div>V18_17 RUNTIME CONTRACT</div><span>EXACT</span></div>
              <dl>
                <div><dt>Inference limit</dt><dd>{targetFps} FPS / 100 ms</dd></div>
                <div><dt>Taxonomy</dt><dd>{classNames.length} classes</dd></div>
                <div><dt>Feature vector</dt><dd>76 dimensions</dd></div>
                <div><dt>Temporal filter</dt><dd>EMA α {health?.config.ema_alpha ?? 0.65}</dd></div>
                <div><dt>Confidence floor</dt><dd>{percent(health?.config.confidence_floor ?? 0.70)}</dd></div>
                <div><dt>Follow Object</dt><dd>Palm → Fist → Palm</dd></div>
              </dl>
            </article>

            <article className="panel server-card">
              <div className="panel-header"><div>LOCAL ENDPOINTS</div><span>PRIVATE TO THIS PC</span></div>
              <div className="endpoint-list"><code>Dashboard  http://127.0.0.1:3000</code><code>API        http://127.0.0.1:8000</code><code>API docs   http://127.0.0.1:8000/docs</code><code>WebSocket  ws://127.0.0.1:8000/ws/live</code></div>
            </article>
          </div>
        </>}
      </section>
    </main>
  );
}
