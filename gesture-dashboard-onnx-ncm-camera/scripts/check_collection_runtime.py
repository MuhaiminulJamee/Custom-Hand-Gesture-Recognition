"""Compare runtime changes on retained camera cases and public feature data.

Writes local diagnostic evidence, never trains or promotes a classifier.
"""
from pathlib import Path
import json
import subprocess
import types
from collections import Counter

import numpy as np

from backend.config import load_runtime_config
from backend.model_runtime import InferenceEngine, RuntimeSession
from research.v18_20 import load_npz, predict_onnx, evaluate


def previous_module(name, path):
    source = subprocess.check_output(['git', 'show', f'HEAD:gesture-dashboard-onnx-ncm-camera/{path}'], text=True)
    module = types.ModuleType(f'backend.{name}')
    module.__package__ = 'backend'
    import sys
    sys.modules[module.__name__] = module
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


def main():
    import cv2
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[1]
    config = load_runtime_config(root / 'models')
    old_geometry = previous_module('previous_geometry', 'backend/geometry.py')
    old_detector = previous_module('previous_detector', 'backend/hand_detection.py')
    old_runtime = previous_module('previous_model_runtime', 'backend/model_runtime.py')
    old_runtime.GeometryResolver = old_geometry.GeometryResolver
    old_runtime.HandDetector = old_detector.HandDetector
    report = {'note': 'Development regressions, not independent customer accuracy.', 'camera': {}}
    cases = {
        'background': [root / '.runtime/current-detection-case.jpg'] * 30,
        'left_retained': sorted((root / '.runtime/left_distance_case').glob('*.jpg')),
        'dorsal_retained': [root / '.runtime/dorsal-review.jpg'] * 20,
    }
    for name, engine_class, session_class in [('before', old_runtime.InferenceEngine, old_runtime.RuntimeSession), ('after', InferenceEngine, RuntimeSession)]:
        engine = engine_class(config, root / 'models')
        for case, paths in cases.items():
            if not paths or not paths[0].exists():
                continue
            session = session_class(config)
            rows = []
            import time
            original_clock = time.monotonic
            try:
                for i, path in enumerate(paths):
                    time.monotonic = lambda i=i: 1000. + i * .1
                    r = engine.process_frame(path.read_bytes(), session, center_crop_ratio=.92)
                    rows.append(dict(prediction=r.get('runtime_prediction'), action=r.get('runtime_action'),
                                     reason=r.get('action_reason', r.get('message')),
                                     ms=r['timing']['total_ms'], mass=r.get('known_gesture_mass'),
                                     detected=bool(r.get('landmarks'))))
            finally:
                time.monotonic = original_clock
            report['camera'][f'{case}_{name}'] = dict(frames=len(rows), predictions=dict(Counter(r['prediction'] for r in rows)),
                detected=sum(r['detected'] for r in rows), actions=dict(Counter(r['action'] for r in rows)),
                p95_ms=float(np.quantile([r['ms'] for r in rows], .95)), reasons=dict(Counter(r['reason'] for r in rows)))
        engine.close()
    data = load_npz(root / 'artifacts/v18_20/public_test.npz')
    p, mass = predict_onnx(root / 'models/gesture_mlp_production.onnx', data['X'])
    import research.v18_20 as evaluation
    current_resolver = evaluation.GeometryResolver
    try:
        evaluation.GeometryResolver = old_geometry.GeometryResolver
        before = evaluate(p, mass, data, config=config)
        evaluation.GeometryResolver = current_resolver
        after = evaluate(p, mass, data, config=config)
    finally:
        evaluation.GeometryResolver = current_resolver
    report['public_test'] = {'before': before['metrics'], 'after': after['metrics'],
                             'after_per_class': after['report']}
    destination = root / '.runtime/collection-runtime-check.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
