import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES    = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'organic', 'e-waste', 'hazardous']
IMG_SIZE       = (224, 224)
RANDOM_SEED    = 42
TRAIN_SPLIT    = 0.70
VAL_SPLIT      = 0.15   # of total; test gets the rest
RAW_DIR        = Path('data/raw')
PROCESSED_DIR  = Path('data/processed')
ALLOWED_EXT    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Hamming distance threshold for near-duplicate pHash (0 = identical, 10 = very similar)
PHASH_THRESHOLD = 8

# Alias map: maps non-standard folder names → canonical class name
ALIASES = {
    'cardboard': 'cardboard',
    'carton':    'cardboard',
    'glass':     'glass',
    'metal':     'metal',
    'aluminum':  'metal',
    'tin':       'metal',
    'paper':     'paper',
    'plastic':   'plastic',
    'trash':     'trash',
    'garbage':   'trash',
    'other':     'trash',
    'general':   'trash',
    'organic':   'organic',
    'food':      'organic',
    'compost':   'organic',
    'e-waste':   'e-waste',
    'ewaste':    'e-waste',
    'electronic':'e-waste',
    'hazardous': 'hazardous',
    'battery':   'hazardous',
    'batteries': 'hazardous',
    'chemical':  'hazardous',
}


def resolve_class(folder_name: str) -> str | None:
    key = folder_name.lower().strip()
    return ALIASES.get(key)


# ── Label Verification ────────────────────────────────────────────────────────

def collect_source_images(source: Path) -> dict[str, list[Path]]:
    """
    Walk source directory and map images to canonical class names.
    Handles flat layout (source/cardboard/*.jpg) and
    nested layout (source/TRAIN/cardboard/*.jpg + source/TEST/cardboard/*.jpg).
    Warns about unrecognized folder names instead of silently skipping them.
    """
    found = {c: [] for c in CLASS_NAMES}
    unrecognized = []

    def scan_dir(d: Path):
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            cls = resolve_class(sub.name)
            if cls:
                imgs = [f for f in sub.iterdir() if f.suffix.lower() in ALLOWED_EXT]
                found[cls].extend(imgs)
                print(f"  {sub.relative_to(source)} → {cls}: {len(imgs)} images")
            else:
                # Could be a split folder (TRAIN/TEST) — recurse one level
                inner_dirs = [x for x in sub.iterdir() if x.is_dir()]
                if inner_dirs:
                    scan_dir(sub)
                else:
                    # Leaf folder with no class match — flag it
                    unrecognized.append(sub.relative_to(source))

    scan_dir(source)

    # ── Label verification report ──────────────────────────────────────────
    if unrecognized:
        print(f"\n{'!'*60}")
        print("LABEL VERIFICATION — UNRECOGNIZED FOLDERS (images skipped):")
        for folder in unrecognized:
            print(f"  '{folder}'  ← not mapped to any class")
        print(f"\nKnown aliases: {sorted(ALIASES.keys())}")
        print(f"Add an entry to ALIASES in merge_data.py to include these.")
        print(f"{'!'*60}\n")
    else:
        print("  Label verification: all folder names recognized.")

    return found


# ── Duplicate Detection ───────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _phash(path: Path) -> imagehash.ImageHash | None:
    try:
        return imagehash.phash(Image.open(path).convert('RGB'))
    except Exception:
        return None


def build_existing_hash_index() -> tuple[set[str], list[imagehash.ImageHash]]:
    """Build MD5 and pHash index of all images already in data/raw/."""
    print("\nBuilding hash index of existing images in data/raw/ ...")
    md5_set: set[str] = set()
    phash_list: list[imagehash.ImageHash] = []

    for cls in CLASS_NAMES:
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() not in ALLOWED_EXT:
                continue
            md5_set.add(_md5(f))
            ph = _phash(f)
            if ph is not None:
                phash_list.append(ph)

    print(f"  Indexed {len(md5_set)} existing images.")
    return md5_set, phash_list


def filter_duplicates(
    new_images: dict[str, list[Path]],
    existing_md5: set[str],
    existing_phashes: list[imagehash.ImageHash],
) -> dict[str, list[Path]]:
    """
    Remove incoming images that are exact or near-duplicates of existing ones.
    Returns filtered dict with only genuinely new images.
    """
    print("\nRunning duplicate detection on incoming images...")
    filtered: dict[str, list[Path]] = {c: [] for c in CLASS_NAMES}

    total_exact = 0
    total_near  = 0
    total_kept  = 0

    for cls, paths in new_images.items():
        for path in paths:
            # 1. Exact duplicate via MD5
            md5 = _md5(path)
            if md5 in existing_md5:
                total_exact += 1
                continue

            # 2. Near-duplicate via pHash
            ph = _phash(path)
            if ph is not None:
                near_dup = any(
                    (ph - existing_ph) <= PHASH_THRESHOLD
                    for existing_ph in existing_phashes
                )
                if near_dup:
                    total_near += 1
                    continue

            # Passed both checks — keep it and add to running index
            existing_md5.add(md5)
            if ph is not None:
                existing_phashes.append(ph)
            filtered[cls].append(path)
            total_kept += 1

    print(f"  Exact duplicates removed  : {total_exact}")
    print(f"  Near-duplicates removed   : {total_near}  (pHash threshold ≤ {PHASH_THRESHOLD})")
    print(f"  Images kept               : {total_kept}")
    return filtered


# ── Copy & Preprocess ─────────────────────────────────────────────────────────

def copy_to_raw(new_images: dict[str, list[Path]]):
    """Copy new images into data/raw/<class>/ with collision-safe filenames."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_added = 0
    for cls, paths in new_images.items():
        if not paths:
            continue
        cls_dir = RAW_DIR / cls
        cls_dir.mkdir(exist_ok=True)
        existing = list(cls_dir.glob('*'))
        start_idx = len(existing)
        for i, src in enumerate(paths):
            dst = cls_dir / f"merged_{start_idx + i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
        print(f"  {cls:12s}: +{len(paths)} images (total now: {start_idx + len(paths)})")
        total_added += len(paths)
    return total_added


def load_and_preprocess():
    """Load all images from data/raw, resize, normalise → [0,1], save .npy files."""
    print("\nLoading and preprocessing all images...")
    images, labels = [], []

    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} does not exist — skipping")
            continue
        cls_files = [f for f in cls_dir.iterdir() if f.suffix.lower() in ALLOWED_EXT]
        print(f"  {cls:12s}: {len(cls_files)} images")
        for img_path in cls_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(label_idx)
            except Exception as e:
                print(f"    skip {img_path.name}: {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    y_onehot = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float32)
    y_onehot[np.arange(len(labels)), labels] = 1.0

    print(f"\nTotal images: {len(images)}")
    return images, y_onehot, labels


def _to_phash(img: np.ndarray) -> imagehash.ImageHash:
    uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return imagehash.phash(Image.fromarray(uint8))


def remove_cross_split_leakage(X_train, y_train, X_val, y_val, X_test, y_test, phash_threshold=PHASH_THRESHOLD):
    """Remove val/test images that are near-duplicates of any train image."""
    print("\nChecking for cross-split leakage (train ↔ val/test)...")

    print("  Building pHash index from train set...")
    train_phashes = [_to_phash(img) for img in tqdm(X_train, desc="  Hashing train", leave=False)]

    def filter_split(X, y, name):
        keep = []
        leaked = 0
        for idx, img in enumerate(tqdm(X, desc=f"  Checking {name}", leave=False)):
            ph = _to_phash(img)
            if any((ph - t) <= phash_threshold for t in train_phashes):
                leaked += 1
            else:
                keep.append(idx)
        print(f"  {name:5s}: {leaked} leaking images removed → {len(keep)} kept")
        return X[keep], y[keep]

    X_val,  y_val  = filter_split(X_val,  y_val,  "val")
    X_test, y_test = filter_split(X_test, y_test, "test")
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_and_save(images, y_onehot, labels_int):
    """Stratified split, leakage removal, and save to data/processed/."""
    test_size  = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    val_size_of_trainval = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)

    X_tv, X_test, y_tv, y_test, l_tv, _ = train_test_split(
        images, y_onehot, labels_int,
        test_size=test_size, random_state=RANDOM_SEED, stratify=labels_int
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_size_of_trainval, random_state=RANDOM_SEED, stratify=l_tv
    )

    print(f"\nSplit → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Remove cross-split leakage
    X_train, y_train, X_val, y_val, X_test, y_test = remove_cross_split_leakage(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"After leakage removal → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / 'X_train.npy', X_train)
    np.save(PROCESSED_DIR / 'y_train.npy', y_train)
    np.save(PROCESSED_DIR / 'X_val.npy',   X_val)
    np.save(PROCESSED_DIR / 'y_val.npy',   y_val)
    np.save(PROCESSED_DIR / 'X_test.npy',  X_test)
    np.save(PROCESSED_DIR / 'y_test.npy',  y_test)
    print("Saved → data/processed/")

    print("\nClass distribution (train):")
    train_ints = np.argmax(y_train, axis=1)
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:12s}: {(train_ints == i).sum()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Merge new dataset into WASP training data')
    parser.add_argument('--source', required=True,
                        help='Path to folder containing class sub-folders')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be merged without copying anything')
    parser.add_argument('--phash-threshold', type=int, default=PHASH_THRESHOLD,
                        help=f'Hamming distance for near-duplicate detection (default: {PHASH_THRESHOLD})')
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"ERROR: source path does not exist: {source}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("WASP — Data Merge Pipeline")
    print(f"{'='*60}")
    print(f"Source : {source}")
    print(f"Target : {RAW_DIR.resolve()}\n")

    print("Current raw image counts:")
    for cls in CLASS_NAMES:
        d = RAW_DIR / cls
        n = len(list(d.glob('*'))) if d.exists() else 0
        print(f"  {cls:12s}: {n}")

    print(f"\nScanning source for images...")
    new_images = collect_source_images(source)

    total_new = sum(len(v) for v in new_images.values())
    if total_new == 0:
        print("\nNo matching images found. Check folder names match class names.")
        print(f"Expected (case-insensitive): {CLASS_NAMES}")
        sys.exit(1)

    print(f"\nFound {total_new} candidate images:")
    for cls, paths in new_images.items():
        if paths:
            print(f"  {cls:12s}: +{len(paths)}")

    # Build hash index and filter duplicates
    existing_md5, existing_phashes = build_existing_hash_index()
    new_images = filter_duplicates(new_images, existing_md5, existing_phashes)

    total_clean = sum(len(v) for v in new_images.values())
    if total_clean == 0:
        print("\nAll incoming images are duplicates of existing data. Nothing to add.")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would copy {total_clean} new images (after deduplication). No files copied.")
        return

    confirm = input(f"\nProceed? This will copy {total_clean} images and rebuild processed data. [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    print("\nCopying images to data/raw/...")
    copy_to_raw(new_images)

    images, y_onehot, labels_int = load_and_preprocess()
    split_and_save(images, y_onehot, labels_int)

    print(f"\n{'='*60}")
    print("Done! Now retrain with:")
    print("  python train_fixed.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
