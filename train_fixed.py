"""
Two-phase EfficientNetB0 training script.

Phase 1 — freeze base, train head only (10 epochs, LR 1e-3)
Phase 2 — unfreeze all, fine-tune (40 epochs max, LR 1e-4, patience 20)

Run from project root:
    python train_fixed.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

CLASS_NAMES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'organic', 'e-waste', 'hazardous']
MODELS_DIR   = Path('models')
CHECKPOINT_DIR = MODELS_DIR / 'checkpoints'
FINAL_DIR      = MODELS_DIR / 'final'
LOG_DIR        = Path('logs')

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val   = np.load('data/processed/X_val.npy')
y_val   = np.load('data/processed/y_val.npy')
X_test  = np.load('data/processed/X_test.npy')
y_test  = np.load('data/processed/y_test.npy')

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
print(f"Pixel range: {X_train.min():.3f} – {X_train.max():.3f}")

# ── Class weights ─────────────────────────────────────────────────────────────
y_int    = np.argmax(y_train, axis=1)
weights  = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weight = dict(enumerate(weights))
print("\nClass weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:12s}: {class_weight[i]:.3f}")

# ── Build model ───────────────────────────────────────────────────────────────
# EfficientNetB0 expects pixels in [0, 255]; our data is [0, 1]
# Rescaling(255) converts correctly.
print("\nBuilding EfficientNetB0 model...")

base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

inputs  = keras.Input(shape=(224, 224, 3))
x       = layers.Rescaling(scale=255.0)(inputs)      # [0,1] → [0,255]
x       = base(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

model = keras.Model(inputs, outputs, name='efficientnetb0_waste')
print(f"Total params: {model.count_params():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Freeze base, train head only
# Lets the randomly-initialised head stabilise before touching pretrained weights
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1 — Head warm-up  (base frozen, 10 epochs, LR 1e-3)")
print("="*60)

base.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

phase1_callbacks = [
    keras.callbacks.CSVLogger(str(LOG_DIR / 'train_phase1_log.csv'))
]

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10,
    class_weight=class_weight,
    callbacks=phase1_callbacks,
    verbose=1
)

best_phase1 = max(history1.history['val_accuracy'])
print(f"\nPhase 1 best val_accuracy: {best_phase1*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Unfreeze all, fine-tune at low LR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2 — Full fine-tune  (all layers, 40 epochs max, LR 1e-4)")
print("="*60)

base.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

phase2_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_DIR / 'efficientnetb0_best.keras'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.CSVLogger(str(LOG_DIR / 'train_phase2_log.csv'))
]

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=40,
    class_weight=class_weight,
    callbacks=phase2_callbacks,
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

best_val = max(history2.history['val_accuracy'])
best_epoch = history2.history['val_accuracy'].index(best_val) + 1
print(f"\nBest val accuracy (phase 2): {best_val*100:.2f}%  (epoch {best_epoch})")
print(f"Epochs run (phase 2)       : {len(history2.history['accuracy'])}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(str(FINAL_DIR / 'efficientnetb0_final.keras'))
print("\n✅  Model saved → models/final/efficientnetb0_final.keras")
print("✅  Best checkpoint → models/checkpoints/efficientnetb0_best.keras")
