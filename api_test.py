"""
Systematic API test — runs all 817 test images through /api/predict
and compares predictions against ground-truth labels.

Usage:
    python api_test.py [--host http://localhost:8090] [--output results/api_test_results.csv]
"""

import argparse
import io
import json
import time
import numpy as np
import requests
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic',
               'trash', 'organic', 'e-waste', 'hazardous']

PROCESSED_DIR = Path('data/processed')
RESULTS_DIR   = Path('results')


def load_test_data():
    X_test = np.load(PROCESSED_DIR / 'X_test.npy')
    y_test = np.load(PROCESSED_DIR / 'y_test.npy')
    labels = np.argmax(y_test, axis=1)
    print(f"Loaded {len(X_test)} test images, {len(CLASS_NAMES)} classes")
    return X_test, labels


def numpy_to_jpeg_bytes(img_array: np.ndarray) -> bytes:
    uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
    pil_img = Image.fromarray(uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def predict_via_api(host: str, img_bytes: bytes, idx: int):
    try:
        resp = requests.post(
            f"{host}/api/predict",
            files={'image': (f'test_{idx:04d}.jpg', img_bytes, 'image/jpeg')},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        print(f"\n  [ERROR] Image {idx}: {e}")
        return None


def run_tests(host: str, output_path: Path):
    X_test, true_labels = load_test_data()
    n = len(X_test)

    results = []
    correct = 0
    errors  = 0

    # Per-class tracking
    class_total   = {c: 0 for c in CLASS_NAMES}
    class_correct = {c: 0 for c in CLASS_NAMES}
    confusion     = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    print(f"\nSending {n} images to {host}/api/predict ...\n")
    start_time = time.time()

    for idx in tqdm(range(n), desc="Testing", unit="img"):
        img_bytes  = numpy_to_jpeg_bytes(X_test[idx])
        true_idx   = int(true_labels[idx])
        true_label = CLASS_NAMES[true_idx]

        response = predict_via_api(host, img_bytes, idx)

        if response is None:
            errors += 1
            results.append({
                'image_index': idx,
                'true_label': true_label,
                'predicted_label': 'ERROR',
                'confidence': 0.0,
                'correct': False,
                'response_time_ms': 0
            })
            continue

        pred        = response.get('prediction', {})
        pred_label  = pred.get('class', 'unknown')
        confidence  = pred.get('confidence', 0.0)
        resp_time   = 0  # api/predict does not return inference_time_ms

        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1

        class_total[true_label] += 1
        if is_correct:
            class_correct[true_label] += 1

        pred_idx = CLASS_NAMES.index(pred_label) if pred_label in CLASS_NAMES else -1
        if pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

        results.append({
            'image_index': idx,
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': round(confidence, 4),
            'correct': is_correct,
            'response_time_ms': resp_time
        })

    elapsed = time.time() - start_time
    tested  = n - errors
    accuracy = correct / tested if tested > 0 else 0

    # ── Save CSV ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved → {output_path}")

    # ── Print Report ──────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("API TEST REPORT — ALL 817 TEST IMAGES")
    print("="*62)
    print(f"  Total images     : {n}")
    print(f"  Successful calls : {tested}")
    print(f"  API errors       : {errors}")
    print(f"  Correct          : {correct}")
    print(f"  Overall Accuracy : {accuracy*100:.2f}%")
    ms_per_img = (elapsed / tested * 1000) if tested > 0 else 0
    print(f"  Total time       : {elapsed:.1f}s  ({ms_per_img:.0f} ms/image)")
    print()

    print(f"  {'Class':<12} {'Total':>6} {'Correct':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for i, cls in enumerate(CLASS_NAMES):
        total   = class_total[cls]
        corr    = class_correct[cls]
        recall  = corr / total if total > 0 else 0
        # Precision = TP / (TP + FP) = confusion[i][i] / sum(confusion[:][i])
        tp_fp   = confusion[:, i].sum()
        prec    = confusion[i][i] / tp_fp if tp_fp > 0 else 0
        print(f"  {cls:<12} {total:>6} {corr:>8} {prec*100:>9.1f}% {recall*100:>7.1f}%")

    print()
    print(f"  Confusion matrix saved in results along with per-image CSV.")
    print("="*62)

    # ── Save confusion matrix ─────────────────────────────────────────────────
    cm_path = RESULTS_DIR / 'api_confusion_matrix.csv'
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + CLASS_NAMES)
        for i, cls in enumerate(CLASS_NAMES):
            writer.writerow([cls] + list(confusion[i]))
    print(f"  Confusion matrix → {cm_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Systematic API test of all test images')
    parser.add_argument('--host',   default='http://localhost:8090',
                        help='Flask API host (default: http://localhost:8090)')
    parser.add_argument('--output', default='results/api_test_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{args.host}/api/health", timeout=5)
        info = r.json()
        print(f"Server healthy — model: {info.get('model_loaded')}, "
              f"classes: {info.get('num_classes')}")
    except Exception as e:
        print(f"ERROR: Cannot reach {args.host} — {e}")
        print("Start the server first:  python app.py")
        return

    run_tests(args.host, Path(args.output))


if __name__ == '__main__':
    main()
