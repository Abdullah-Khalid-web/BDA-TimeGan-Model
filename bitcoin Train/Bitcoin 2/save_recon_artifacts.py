# src/save_recon_artifacts.py
import os
import pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery

# =============================
# PERFORMANCE SETTINGS
# =============================
tf.config.optimizer.set_jit(True)

try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print("Mixed precision enabled")
except Exception:
    pass

# =============================
# CONFIG
# =============================
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/recon"

SEQ_LEN = 168
HIDDEN_DIM = 128      # ✅ must match training!
NUM_LAYERS = 3       # ✅ must match training!

# =============================
# LOAD FEATURE DIM SAFELY
# =============================
train_ref = np.load(
    os.path.join(DATA_DIR, "train.npy"),
    mmap_mode="r"
)

feature_dim = train_ref.shape[2]

print("Feature dim:", feature_dim)

# =============================
# MODEL REBUILD
# =============================
embedder = Embedder(
    input_dim=feature_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

recovery = Recovery(
    hidden_dim=HIDDEN_DIM,
    output_dim=feature_dim,
    num_layers=NUM_LAYERS
)

# Build model shapes
_ = embedder(tf.zeros([1, SEQ_LEN, feature_dim]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# =============================
# RESTORE CHECKPOINT
# =============================
ckpt = tf.train.Checkpoint(
    embedder=embedder,
    recovery=recovery
)

manager = tf.train.CheckpointManager(
    ckpt,
    CKPT_DIR,
    max_to_keep=5
)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("✅ Restored checkpoint:", manager.latest_checkpoint)
else:
    raise SystemExit(f"❌ No checkpoint found in {CKPT_DIR}. Train first.")

# =============================
# SAVE FINAL .WEIGHTS FILES
# =============================
embedder_path = os.path.join(CKPT_DIR, "embedder_final.weights.h5")
recovery_path = os.path.join(CKPT_DIR, "recovery_final.weights.h5")

embedder.save_weights(embedder_path)
recovery.save_weights(recovery_path)

print("✅ Final weights saved")

# =============================
# LOAD TEST SAMPLES SAFELY
# =============================
test = np.load(
    os.path.join(DATA_DIR, "test.npy"),
    mmap_mode="r"
)

examples = test[:5]   # ✅ small sample only

# =============================
# RECONSTRUCT
# =============================
h_ex = embedder(examples, training=False)
x_rec = recovery(h_ex, training=False)

x_rec = tf.cast(x_rec, tf.float32).numpy()

# =============================
# LOAD SCALER
# =============================
scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
if not os.path.exists(scalers_path):
    scalers_path = os.path.join(DATA_DIR, "scaler.pkl")

with open(scalers_path, "rb") as f:
    scaler = pickle.load(f)

# =============================
# INVERSE SCALE CORRECTLY
# =============================
n, T, D = x_rec.shape

x_rec_inv   = np.zeros_like(x_rec, dtype=np.float32)
x_orig_inv = np.zeros_like(examples, dtype=np.float32)

if isinstance(scaler, list):
    # Per-feature scalers (correct reshape!)
    for j, sc in enumerate(scaler):
        x_rec_inv[..., j] = sc.inverse_transform(
            x_rec[..., j].reshape(-1, 1)
        ).reshape(n, T)

        x_orig_inv[..., j] = sc.inverse_transform(
            examples[..., j].reshape(-1, 1)
        ).reshape(n, T)
else:
    # One global scaler
    x_rec_inv = scaler.inverse_transform(
        x_rec.reshape(-1, D)
    ).reshape(n, T, D)

    x_orig_inv = scaler.inverse_transform(
        examples.reshape(-1, D)
    ).reshape(n, T, D)

# =============================
# SAVE ARTIFACTS
# =============================
np.save(os.path.join(CKPT_DIR, "recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(CKPT_DIR, "recon_examples_rec.npy"),  x_rec_inv)

# =============================
# DIAGNOSTICS
# =============================
mae = float(np.mean(np.abs(x_orig_inv - x_rec_inv)))

print("✅ Reconstruction complete")
print("Artifacts saved to:", CKPT_DIR)
print(f"Mean Absolute Error (all samples): {mae:.6f}")
