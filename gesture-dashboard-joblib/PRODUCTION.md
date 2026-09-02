# Production operations guide

This guide covers the hardened Windows PC deployment. The classifier's
`no_gesture` quality improvement is explicitly deferred; its metrics remain
visible, but it is excluded from the frozen release gate until the dataset work
is complete. No class, resolver, EMA threshold, or Follow Object behavior is
changed by the production controls.

## Release policy

A deployable PC release must satisfy all of the following:

1. Every runtime artifact matches `models/gesture_artifact_manifest.json`.
2. Runtime configuration has the exact 15-class order and 76-D feature order.
3. MediaPipe and the selected classifier pass the startup smoke test.
4. The untouched-test macro-F1, excluding deferred classes, is at least 0.90.
5. Every non-deferred class has F1 of at least 0.80.
6. The target PC passes an extended 10 FPS live-camera test with acceptable
   p95/p99 latency and error rate.

The frozen report is `models/gesture_production_qualification.json`. Regenerate
it after every trusted model import or accepted online update:

```powershell
.\.venv\Scripts\python.exe .\scripts\qualify_release.py --require-pass
.\.venv\Scripts\python.exe .\scripts\build_artifact_manifest.py
```

The order matters: qualify the final model first, then sign the final runtime
artifact set.

## Safe defaults

`start_dashboard.bat` enables production mode with these defaults:

- Services bind only to `127.0.0.1`.
- Artifact signatures are required before Joblib models are loaded.
- Force Learn is disabled.
- hot artifact reload and rollback APIs are disabled.
- only four concurrent camera sessions are allowed.
- image and WebSocket frame sizes are bounded.
- rolling runtime metrics retain the most recent 1,000 frames.

Safe Learn remains available and validates every candidate against the preserved
validation cache. It writes a rollback checkpoint before replacing the stored
OnlineMLP candidate. The last qualified model remains live in production.
Accepted PC updates make the qualification report stale by design; qualification
must be rerun before the best passing candidate is promoted as a new release.

## Administrative maintenance

Force Learn is not a normal production feature. If it is temporarily required in
an isolated maintenance session, stop the dashboard and set all three variables
in the same PowerShell window before starting the services manually:

```powershell
$env:GESTURE_ALLOW_FORCE_LEARNING = "true"
$env:GESTURE_ALLOW_ARTIFACT_RELOAD = "true"
$env:GESTURE_ADMIN_TOKEN = "use-a-long-random-secret"
```

Never expose port 8000 to another computer while maintenance mode is enabled.
After maintenance, rerun qualification, rebuild the manifest, clear the variables,
and restart normally.

Backups are stored in `models/online_backups_pc`. The rollback API accepts only a
file name returned by `GET /api/online-learning/backups` and requires the
`X-Admin-Token` header when administrative operations are enabled.

## Monitoring and logs

- `GET /livez` — process liveness.
- `GET /readyz` — model, MediaPipe, and artifact-integrity readiness.
- `GET /api/health` — complete operational status.
- `GET /api/runtime-metrics` — rolling JSON metrics.
- `GET /metrics` — Prometheus-compatible counters and latency quantiles.
- `.runtime/logs/backend.jsonl` — rotating structured backend log.

Monitor at minimum:

- pipeline p50, p95, and p99 latency;
- 10 FPS frame-budget pass rate;
- inference error rate;
- unexpected WebSocket disconnects;
- action count during idle/no-command sessions;
- model fingerprint and qualification freshness.

The browser keeps only one frame in flight, so overload cannot create an
unbounded inference queue.

## Incident response

If commands become unreliable after an update:

1. Stop the camera and dashboard.
2. Preserve `.runtime/logs`, `feedback`, and `models/gesture_online_events_pc.csv`.
3. Start an authenticated maintenance session.
4. Restore the most recent known-good `state_before_*.joblib` backup.
5. Rerun the frozen qualification and artifact manifest scripts.
6. Run automated tests and a live 10 FPS soak test before returning to service.

If artifact verification fails, do not regenerate the manifest blindly. Re-import
the known trusted Drive export, qualify it, and then create a new manifest.

## Runtime boundary

This edition is intentionally limited to trusted Python/Joblib classifiers. Other
deployment formats and their handover procedures belong in separate projects and
are not part of this release.
