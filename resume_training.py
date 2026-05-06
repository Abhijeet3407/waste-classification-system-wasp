"""
Resume Phase 2 from best checkpoint with speed optimisations:
  - Legacy Adam  (runs ~2-3x faster on Apple M-series)
  - Batch size 64 (halves batches per epoch: 120 → 60)
  - Starts directly at Phase 2 — no warmup needed

Run:
    python resume_training.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

CLASS_NAMES    = ['cardboard','glass','metal','paper','plastic','trash','organic','e-waste','hazardous']
CHECKPOINT     = Path('models/checkpoints/efficientnetb0_best.keras')
FINAL_DIR      = Path('models/final')
LOG_DIR        = Path('logs')

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val   = np.load('data/processed/X_val.npy')
y_val   = np.load('data/processed/y_val.npy')
X_test  = np.load('data/processed/X_test.npy')
y_test  = np.load('data/processed/y_test.npy')
print(f"Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

# ── Class weights ─────────────────────────────────────────────────────────────
y_int        = np.argmax(y_train, axis=1)
weights      = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weight = dict(enumerate(weights))
print("\nClass weights:")
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:12s}: {class_weight[i]:.3f}")

# ── Load best checkpoint ──────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CHECKPOINT}")
model = keras.models.load_model(str(CHECKPOINT))
print(f"Checkpoint loaded — {model.count_params():,} params")

# Unfreeze all layers (Phase 2)
for layer in model.layers:
    layer.trainable = True

# ── Compile with legacy Adam (fast on Apple Silicon) ─────────────────────────
model.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(Path('models/checkpoints/efficientnetb0_best.keras')),
        monitor='val_accuracy', mode='max',
        save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20, mode='max',
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        min_lr=1e-7, verbose=1
    ),
    keras.callbacks.CSVLogger(str(LOG_DIR / 'resume_training_log.csv'))
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 RESUME — Legacy Adam | Batch 64 | 40 epochs | patience 20")
print("="*60 + "\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,          # faster: 60 batches/epoch instead of 120
    epochs=40,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)
loss, acc, top3 = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy  : {acc*100:.2f}%")
print(f"Top-3 Accuracy : {top3*100:.2f}%")
print(f"Test Loss      : {loss:.4f}")

best_val = max(history.history['val_accuracy'])
best_ep  = history.history['val_accuracy'].index(best_val) + 1
print(f"\nBest val accuracy : {best_val*100:.2f}%  (epoch {best_ep})")
print(f"Epochs run        : {len(history.history['accuracy'])}")

# ── Save final ────────────────────────────────────────────────────────────────
model.save(str(FINAL_DIR / 'efficientnetb0_final.keras'))
print("\n✅  Model saved → models/final/efficientnetb0_final.keras")
print("✅  Best checkpoint → models/checkpoints/efficientnetb0_best.keras")
