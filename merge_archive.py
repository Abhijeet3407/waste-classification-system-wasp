"""
Merges ~/Downloads/archive dataset into WASP training data.

Category mapping:
  Recyclable/glass_containers          → glass
  Recyclable/cans_all_type             → metal
  Recyclable/paper_products            → paper
  Recyclable/plastic_bottles           → plastic
  Non-Recyclable/platics_bags_wrappers → plastic
  Non-Recyclable/* (rest)              → trash
  Organic/*                            → trash
  Hazardous/*                          → trash

Run:
    python merge_archive.py [--dry-run]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
ARCHIVE      = Path.home() / 'Downloads' / 'archive'
CLASS_NAMES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'organic', 'e-waste', 'hazardous']
IMG_SIZE     = (224, 224)
RANDOM_SEED  = 42
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
RAW_DIR      = Path('data/raw')
PROC_DIR     = Path('data/processed')
ALLOWED_EXT  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# sub-folder path (relative to ARCHIVE) → canonical class
MAPPING = {
    'Recyclable/Recyclable/glass_containers':           'glass',
    'Recyclable/Recyclable/cans_all_type':              'metal',
    'Recyclable/Recyclable/paper_products':             'paper',
    'Recyclable/Recyclable/plastic_bottles':            'plastic',
    'Non-Recyclable/Non-Recyclable/platics_bags_wrappers': 'plastic',
    'Non-Recyclable/Non-Recyclable/ceramic_product':    'trash',
    'Non-Recyclable/Non-Recyclable/sanitary_napkin':    'trash',
    'Non-Recyclable/Non-Recyclable/stroform_product':   'trash',
    'Non-Recyclable/Non-Recyclable/diapers':            'trash',
    'Organic/Organic/egg_shells':                       'trash',
    'Organic/Organic/yard_trimmings':                   'trash',
    'Organic/Organic/kitchen_waste':                    'trash',
    'Organic/Organic/food_scraps':                      'trash',
    'Organic/Organic/coffee_tea_bags':                  'trash',
    'Hazardous/Hazardous/batteries':                    'trash',
    'Hazardous/Hazardous/paints':                       'trash',
    'Hazardous/Hazardous/e-waste':                      'trash',
    'Hazardous/Hazardous/pesticides':                   'trash',
}


def collect_images() -> dict[str, list[Path]]:
    found = {c: [] for c in CLASS_NAMES}
    for rel_path, cls in MAPPING.items():
        folder = ARCHIVE / rel_path
        if not folder.exists():
            print(f"  WARNING: not found — {folder}")
            continue
        imgs = [f for f in folder.iterdir() if f.suffix.lower() in ALLOWED_EXT]
        found[cls].extend(imgs)
        print(f"  {rel_path.split('/')[-1]:35s} → {cls:10s}  ({len(imgs)} images)")
    return found


def copy_to_raw(new_images: dict[str, list[Path]]):
    for cls, paths in new_images.items():
        if not paths:
            continue
        cls_dir = RAW_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        start = len(list(cls_dir.glob('*')))
        for i, src in enumerate(paths):
            dst = cls_dir / f"archive_{start + i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
        print(f"  {cls:12s}: +{len(paths)} copied")


def rebuild_processed():
    print("\nPreprocessing all images in data/raw/ ...")
    images, labels = [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = RAW_DIR / cls
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in ALLOWED_EXT]
        print(f"  {cls:12s}: {len(files)} images")
        for fp in files:
            try:
                img = Image.open(fp).convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(label_idx)
            except Exception as e:
                print(f"    skip {fp.name}: {e}")

    X = np.array(images, dtype=np.float32)
    y_int = np.array(labels, dtype=np.int32)
    y = np.zeros((len(y_int), len(CLASS_NAMES)), dtype=np.float32)
    y[np.arange(len(y_int)), y_int] = 1.0
    print(f"\n  Total: {len(X)} images")

    test_size = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    val_size  = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)

    X_tv, X_te, y_tv, y_te, l_tv, _ = train_test_split(
        X, y, y_int, test_size=test_size, random_state=RANDOM_SEED, stratify=y_int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_size, random_state=RANDOM_SEED, stratify=l_tv)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROC_DIR / 'X_train.npy', X_tr)
    np.save(PROC_DIR / 'y_train.npy', y_tr)
    np.save(PROC_DIR / 'X_val.npy',   X_va)
    np.save(PROC_DIR / 'y_val.npy',   y_va)
    np.save(PROC_DIR / 'X_test.npy',  X_te)
    np.save(PROC_DIR / 'y_test.npy',  y_te)

    print(f"\n  Split → Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
    print("\n  Class distribution (train):")
    tr_int = np.argmax(y_tr, axis=1)
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:12s}: {(tr_int == i).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview mapping without copying files')
    args = parser.parse_args()

    if not ARCHIVE.exists():
        print(f"ERROR: {ARCHIVE} not found.")
        sys.exit(1)

    print("=" * 60)
    print("WASP — Archive Import")
    print("=" * 60)
    print(f"Source: {ARCHIVE}\n")

    print("Scanning archive...")
    new_images = collect_images()
    total = sum(len(v) for v in new_images.values())

    print(f"\nSummary of images to add:")
    for cls in CLASS_NAMES:
        n = len(new_images[cls])
        cur = len(list((RAW_DIR / cls).glob('*'))) if (RAW_DIR / cls).exists() else 0
        print(f"  {cls:12s}: {cur} existing + {n} new = {cur + n}")

    if args.dry_run:
        print(f"\n[dry-run] Would add {total} images. No files copied.")
        return

    print(f"\nThis will copy {total} images and rebuild all processed .npy files.")
    confirm = input("Proceed? [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    print("\nCopying images...")
    copy_to_raw(new_images)

    rebuild_processed()

    print("\n" + "=" * 60)
    print("Done! Now retrain:")
    print("  python train_fixed.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
