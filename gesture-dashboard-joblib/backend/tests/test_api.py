from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_health_exposes_exact_runtime_contract():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["feature_count"] == 76
    assert payload["config"]["target_fps"] == 10.0
    assert len(payload["config"]["class_names"]) == 15
    assert payload["config"]["class_names"][11] == "fist"
    assert payload["config"]["follow_object_sequence"] == ["palm", "fist", "palm"]
    assert payload["config"]["class_names"][-1] == "no_gesture"
    assert payload["engine"]["mediapipe"]["ready"] is True
    assert payload["artifact_integrity"]["verified"] is True
    assert payload["production"]["production_mode"] is True
    assert payload["production"]["allow_force_learning"] is False
    assert payload["production"]["release_qualification"]["deferred_classes"] == ["no_gesture"]
    assert payload["production"]["release_qualification"]["pc_release_ready"] is True
    assert payload["engine"]["model"]["selected_model_name"] == "MLP"
    assert payload["engine"]["model"]["selectable_models"] == ["MLP"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_operational_endpoints_are_available():
    assert client.get("/livez").json() == {"status": "alive"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    metrics = client.get("/api/runtime-metrics")
    assert metrics.status_code == 200
    assert "timing" in metrics.json()
    prometheus = client.get("/metrics")
    assert prometheus.status_code == 200
    assert "gesture_frames_total" in prometheus.text


def test_unknown_uploaded_image_model_is_rejected():
    response = client.post(
        "/api/predict-image?model_name=not-a-model",
        files={"image": ("frame.jpg", b"not-used", "image/jpeg")},
    )
    assert response.status_code == 400


def test_websocket_starts_and_resets_session():
    with client.websocket_connect("/ws/live") as websocket:
        hello = websocket.receive_json()
        assert hello["type"] == "hello"
        websocket.send_json({"type": "reset"})
        assert websocket.receive_json()["type"] == "reset_complete"
        websocket.send_json({"type": "start_follow_object"})
        started = websocket.receive_json()
        assert started["type"] == "follow_object_started"
        assert started["follow_object"]["active"] is True
        assert started["follow_object"]["expected_gesture"] == "palm"
