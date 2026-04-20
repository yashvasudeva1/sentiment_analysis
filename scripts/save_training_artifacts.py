"""
save_training_artifacts.py
──────────────────────────
Generates the two JSON files used by the Streamlit app for
Training Insights and Evaluation visualizations.

Script location:
    scripts/save_training_artifacts.py

Output location:
    artifacts/training_history.json
    artifacts/eval_metrics.json

Assumes:
    - model / history are available from your training notebook / script
    - X_train, y_train, X_test, y_test are already padded & one-hot encoded
"""

import json
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
import tensorflow as tf
from pathlib import Path

LABELS = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'


# ─── 1. Save training history ────────────────────────────────────────────────
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, ...)

def save_history(history_object, path=ARTIFACTS_DIR / 'training_history.json'):
    """Pass the return value of model.fit() directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = {k: [float(v) for v in vals]
            for k, vals in history_object.history.items()}
    with open(path, 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'✅  Saved training history → {path}')


# ─── 2. Save evaluation metrics ──────────────────────────────────────────────
def save_eval_metrics(model, X_train, y_train_oh, X_test, y_test_oh,
                      path=ARTIFACTS_DIR / 'eval_metrics.json'):
    """
    model       : trained Keras model
    X_train     : padded training sequences
    y_train_oh  : one-hot encoded train labels  (shape: [n, 4])
    X_test      : padded test sequences
    y_test_oh   : one-hot encoded test labels   (shape: [n, 4])
    """

    def _metrics_for(X, y_oh, split_name):
        loss, acc = model.evaluate(X, y_oh, verbose=0)
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_oh, axis=1)

        p, r, f, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)

        per_class_p, per_class_r, per_class_f, per_class_s = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        per_class = {
            LABELS[i]: {
                'precision': float(per_class_p[i]),
                'recall':    float(per_class_r[i]),
                'f1':        float(per_class_f[i]),
                'support':   int(per_class_s[i]),
            }
            for i in range(len(LABELS))
        }

        return {
            'accuracy':  float(acc),
            'loss':      float(loss),
            'precision': float(p),
            'recall':    float(r),
            'f1':        float(f),
        }, per_class

    train_overall, train_pc = _metrics_for(X_train, y_train_oh, 'train')
    test_overall,  test_pc  = _metrics_for(X_test,  y_test_oh,  'test')

    output = {
        'train': train_overall,
        'test':  test_overall,
        'per_class': {
            'train': train_pc,
            'test':  test_pc,
        }
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'✅  Saved evaluation metrics → {path}')
    return output


# ─── Example usage (replace with your actual variables) ──────────────────────
if __name__ == '__main__':
    # Uncomment and replace with your actual training objects:
    #
    # history_obj = model.fit(
    #     X_train, y_train, epochs=10,
    #     validation_data=(X_test, y_test), batch_size=64
    # )
    # save_history(history_obj)
    # save_eval_metrics(model, X_train, y_train, X_test, y_test)

    print("Edit this script with your actual model/data variables and run it.")
    print("It will create training_history.json and eval_metrics.json in the current folder.")
