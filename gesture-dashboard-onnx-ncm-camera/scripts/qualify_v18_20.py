"""Freeze validation-selected settings, then evaluate and optionally deploy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from backend.artifact_integrity import ArtifactRegistry
from backend.config import RuntimeConfig
from research.v18_20 import LABELS, load_npz, evaluate, predict_onnx, benchmark


def qualify(root, deploy=False):
    root = Path(root)
    output = root / 'artifacts/v18_20'
    candidate = output / 'candidate.onnx'
    config = RuntimeConfig(known_mass_floor=.95)
    selection = dict(known_mass_floor=.95, confidence_floor=.8, probability_margin_floor=.18,
                     source='Settings fixed using public and legacy validation; test excluded from selection',
                     model_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    evaluations, summaries = {}, {}
    for split, path in [('public_validation', output / 'public_val.npz'), ('legacy_validation', root / 'models/gesture_online_validation_cache.npz'),
                        ('public_test', output / 'public_test.npz'), ('legacy_test', root / 'models/gesture_online_untouched_test_cache.npz')]:
        data = load_npz(path)
        for name, model in [('baseline', output / 'baseline/gesture_mlp_production.onnx'), ('v18_20', candidate)]:
            p, mass = predict_onnx(model, data['X'])
            settings = config if name == 'v18_20' else RuntimeConfig()
            result = evaluate(p, mass, data, settings)
            key = f'{split}_{name}'
            result['metrics']['known_mass_acceptance'] = float(np.mean(mass[data['y'] >= 0] >= settings.known_mass_floor)) if split.startswith('legacy') else float(np.mean(mass[data['y'] < 8] >= settings.known_mass_floor))
            evaluations[key] = (data, result)
            summaries[key] = result['metrics']
            pd.DataFrame(result['report']).T.to_csv(output / f'{key}_classification.csv')
            pd.DataFrame(result['confusion'], index=LABELS, columns=LABELS).to_csv(output / f'{key}_confusion.csv')
            print(key, json.dumps(result['metrics']), flush=True)
    latency = benchmark(candidate, evaluations['public_test_v18_20'][0]['X'])
    summaries['classifier_latency'] = latency
    (output / 'evaluation.json').write_text(json.dumps(summaries, indent=2) + '\n')
    public = summaries['public_test_v18_20']
    legacy = summaries['legacy_test_v18_20']
    passed = bool(public['runtime_macro_f1'] >= .97 and public['accepted_precision'] >= .99
                  and public['unknown_false_accept_rate'] <= .02
                  and legacy['raw_known_accuracy'] >= .985 and legacy['raw_known_macro_f1'] >= .98
                  and legacy['unknown_false_accept_rate'] <= .03 and latency['p95_ms'] < 5)
    (output / 'qualification.json').write_text(json.dumps(dict(passed=passed, selection=selection,
        note='Public test is participant-disjoint within imported public data. Legacy test is a regression set with known historic overlap; neither is a live camera accuracy guarantee.'), indent=2) + '\n')
    if not deploy:
        return summaries
    if not passed:
        raise RuntimeError('Candidate did not meet deployment checks; baseline retained.')
    models = root / 'models'
    metadata = json.loads((output / 'baseline/gesture_mlp_onnx_metadata.json').read_text())
    parity = json.loads(candidate.with_suffix('.parity.json').read_text())
    metadata.update(model_name='v18_20 BalancedMLP', created_utc=datetime.now(timezone.utc).isoformat(),
                    onnx_sha256=selection['model_sha256'], onnx_bytes=candidate.stat().st_size,
                    onnx_ir_version=8, opsets={'ai.onnx': 16}, parity=parity,
                    internal_class_order=LABELS, training_labels_ncm_calibrated=True,
                    runtime_geometry={'dorsal_independent_support': True,
                                      'minimum_unknown_model_hold_seconds': 0.35,
                                      'reviewed_negative_feedback_veto': True},
                    runtime_note='Eight exposed command probabilities. no_gesture is an internal negative training class, not an exposed ninth command; known_gesture_mass preserves its rejection evidence.',
                    public_evaluation=public, training=json.loads((output / 'data_audit.json').read_text()))
    metadata['horizontal_direction_calibration']['source_columns_swapped'] = False
    metadata['horizontal_direction_calibration']['training_labels_ncm_calibrated'] = True
    # Keep preflight quality fields on the historical regression set and record
    # public-test metrics separately, with explicit provenance.
    data, result = evaluations['legacy_test_v18_20']
    p, mass = predict_onnx(candidate, data['X'])
    known = data['y'] >= 0
    report = classification_report(data['y'][known], p[known].argmax(1), labels=np.arange(8), target_names=LABELS[:8], output_dict=True, zero_division=0)
    metadata['quality'] = dict(accuracy=legacy['raw_known_accuracy'], macro_f1=legacy['raw_known_macro_f1'],
        known_acceptance_rate=float(np.mean(mass[known] >= .95)), unknown_false_acceptance_rate=float(np.mean(mass[~known] >= .95)),
        minimum_per_class_f1=min(report[label]['f1-score'] for label in LABELS[:8]), evaluation_source='legacy regression cache; historical feature overlap exists')
    config_data = json.loads((models / 'gesture_mobile_runtime_config.json').read_text())
    config_data['temporal_smoothing']['known_mass_floor'] = .95
    config_data['v18_20'] = dict(training_examples_per_class=4096, internal_classes=9, classifier_threads=1,
                               public_test_runtime_macro_f1=public['runtime_macro_f1'])
    shutil.copy2(candidate, models / 'gesture_mlp_production.onnx')
    (models / 'gesture_mlp_onnx_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (models / 'gesture_mobile_runtime_config.json').write_text(json.dumps(config_data, indent=2) + '\n')
    pd.DataFrame([dict(class_name=label, precision=report[label]['precision'], recall=report[label]['recall'], f1_score=report[label]['f1-score'], support=report[label]['support']) for label in LABELS[:8]]).rename(columns={'class_name': 'class'}).to_csv(models / 'eight_gesture_classification_report.csv', index=False)
    pd.DataFrame(confusion_matrix(data['y'][known], p[known].argmax(1), labels=np.arange(8)), index=LABELS[:8], columns=LABELS[:8]).rename_axis('actual').to_csv(models / 'eight_gesture_confusion_matrix.csv')
    pd.DataFrame([dict(model='v18_20', **metadata['quality'])]).to_csv(models / 'eight_gesture_test_metrics.csv', index=False)
    source_names = list(data['source_class_names'])
    rock = data['source_y'] == source_names.index('rock')
    dorsal = data['source_y'] == source_names.index('dorsal_hand')
    down = data['source_y'] == source_names.index('one_down')
    guessed = p.argmax(1)
    pd.DataFrame([dict(rock_hard_negative_samples=int(rock.sum()),
        rock_false_acceptance_rate=float(np.mean(mass[rock] >= .95)),
        rock_accepted_as_down_rate=float(np.mean((mass[rock] >= .95) & (guessed[rock] == 3))),
        dorsal_samples=int(dorsal.sum()), dorsal_recall=float(np.mean(guessed[dorsal] == 6)),
        dorsal_as_down_rate=float(np.mean(guessed[dorsal] == 3)), down_samples=int(down.sum()),
        down_recall=float(np.mean(guessed[down] == 3)), down_as_dorsal_rate=float(np.mean(guessed[down] == 6)))]).to_csv(models / 'eight_gesture_hard_case_metrics.csv', index=False)
    ArtifactRegistry(models).build()
    print('Deployed qualified v18_20 model; previous model retained in artifacts/v18_20/baseline.', flush=True)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    qualify(Path.cwd(), deploy=args.deploy)
