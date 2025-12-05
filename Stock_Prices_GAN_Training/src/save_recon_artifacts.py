# src/save_recon_artifacts.py
import os, pickle, argparse
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery

# ---------------- Parse Arguments ----------------
parser = argparse.ArgumentParser(description='Save reconstruction artifacts and final weights')
parser.add_argument("--data_dir", default="data/processed/stock_prices",
                    help="Path to processed data directory")
parser.add_argument("--ckpt_dir", default="outputs/checkpoints/recon",
                    help="Path to checkpoint directory")
parser.add_argument("--hidden_dim", type=int, default=64,
                    help="Hidden dimension (must match training)")
parser.add_argument("--num_layers", type=int, default=2,
                    help="Number of layers (must match training)")
args = parser.parse_args()

# ---------------- Configuration ----------------
DATA_DIR = args.data_dir
CKPT_DIR = args.ckpt_dir
SEQ_LEN = 168
HIDDEN_DIM = args.hidden_dim

# ---------------- Model Reconstruction ----------------
feature_dim = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
embedder = Embedder(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_layers=args.num_layers)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=feature_dim, num_layers=args.num_layers)

# Build model shapes
_ = embedder(tf.zeros([1, SEQ_LEN, feature_dim]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ---------------- Checkpoint Loading ----------------
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("Restored from checkpoint:", manager.latest_checkpoint)
else:
    print("⚠️ No checkpoint found in", CKPT_DIR)
    raise SystemExit("Please re-run train_recon_tf.py before saving artifacts.")

# ---------------- Save final .weights.h5 ----------------
embedder_path = os.path.join(CKPT_DIR, "embedder_final.weights.h5")
recovery_path = os.path.join(CKPT_DIR, "recovery_final.weights.h5")
embedder.save_weights(embedder_path)
recovery.save_weights(recovery_path)
print("✅ Saved weights:", embedder_path, "and", recovery_path)

# ---------------- Reconstruction Example Generation ----------------
test = np.load(os.path.join(DATA_DIR, "test.npy"))
examples = test[:3]

# Forward pass
h_ex = embedder(examples, training=False)
x_rec = recovery(h_ex, training=False).numpy()

# Load scaler (support both single and per-feature)
scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
if not os.path.exists(scalers_path):
    scalers_path = os.path.join(DATA_DIR, "scaler.pkl")

with open(scalers_path, "rb") as f:
    scaler = pickle.load(f)

# Inverse scaling
n, T, D = x_rec.shape
x_rec_inv = np.empty_like(x_rec)
x_orig_inv = np.empty_like(examples)

if isinstance(scaler, list):
    # per-feature scalers (FIXED: proper reshaping)
    for j, sc in enumerate(scaler):
        x_rec_inv[:, :, j] = sc.inverse_transform(x_rec[:, :, j].reshape(-1, 1)).reshape(n, T)
        x_orig_inv[:, :, j] = sc.inverse_transform(examples[:, :, j].reshape(-1, 1)).reshape(n, T)
else:
    # single scaler
    x_rec_inv = scaler.inverse_transform(x_rec.reshape(-1, D)).reshape(n, T, D)
    x_orig_inv = scaler.inverse_transform(examples.reshape(-1, D)).reshape(n, T, D)

# Save reconstruction results
np.save(os.path.join(CKPT_DIR, "recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(CKPT_DIR, "recon_examples_rec.npy"), x_rec_inv)

# ---------------- Diagnostics ----------------
mae = float(np.mean(np.abs(x_orig_inv - x_rec_inv)))
print("✅ Reconstruction complete.")
print("Reconstruction examples saved to:", CKPT_DIR)
print(f"Mean Absolute Error (all features, all samples): {mae:.6f}")

# Per-feature MAE
print("\nPer-feature Mean Absolute Error:")
feature_names = ['open', 'high', 'low', 'close', 'volume'] if D == 5 else [f'feature_{i}' for i in range(D)]
for j in range(D):
    mae_feat = float(np.mean(np.abs(x_orig_inv[:, :, j] - x_rec_inv[:, :, j])))
    print(f"  {feature_names[j]}: {mae_feat:.6f}")