# src/check_training_results.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

# Configuration
SEQ_LEN = 168
FEATURE_DIM = 14
HIDDEN_DIM = 128
Z_DIM = 32
CKPT_DIR = "outputs/checkpoints/timegan_wgan_gp_fixed"

# Load real data for comparison
DATA_DIR = "data/processed/crypto"
real_data = np.load(os.path.join(DATA_DIR, "train.npy"))

# Build and restore model
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=1)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=2)

# Build models
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# Restore checkpoint
ckpt = tf.train.Checkpoint(generator=generator, supervisor=supervisor, recovery=recovery)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=6)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(f"✅ Restored from: {manager.latest_checkpoint}")
else:
    print("❌ No checkpoint found")
    exit()

# Generate sample data
print("🔍 Generating diagnostic samples...")
Z = tf.random.normal([10, SEQ_LEN, Z_DIM])
Ehat = generator(Z, training=False)
Hhat = supervisor(Ehat, training=False)
Xhat = recovery(Hhat, training=False).numpy()

print(f"📊 Synthetic data shape: {Xhat.shape}")
print(f"📊 Real data shape: {real_data.shape}")

# Basic statistics comparison
print("\n=== STATISTICS COMPARISON ===")
print("Feature | Real Mean | Synth Mean | Real Std | Synth Std")
print("-" * 55)

for i in range(min(6, FEATURE_DIM)):  # Show first 6 features
    real_stats = real_data[:, :, i].flatten()
    synth_stats = Xhat[:, :, i].flatten()
    
    print(f"F{i:02d}    | {np.mean(real_stats):8.4f} | {np.mean(synth_stats):10.4f} | {np.std(real_stats):8.4f} | {np.std(synth_stats):8.4f}")

# Check for common issues
print("\n=== DIAGNOSTIC CHECKS ===")
print(f"NaN values in synthetic: {np.isnan(Xhat).sum()}")
print(f"Inf values in synthetic: {np.isinf(Xhat).sum()}")
print(f"Synthetic data range: [{Xhat.min():.4f}, {Xhat.max():.4f}]")
print(f"Real data range: [{real_data.min():.4f}, {real_data.max():.4f}]")

# Check gradient norms (if possible during training)
print(f"\n=== MODEL ANALYSIS ===")
print(f"Generator trainable variables: {len(generator.trainable_variables)}")
print(f"Supervisor trainable variables: {len(supervisor.trainable_variables)}")
print(f"Recovery trainable variables: {len(recovery.trainable_variables)}")