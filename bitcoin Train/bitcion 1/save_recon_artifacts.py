# src/save_recon_artifacts.py
import os, pickle, json
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery

# ---------------- Configuration Matching Your Original Training ----------------
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/recon"  # Use original directory, not enhanced

# Auto-detect dimensions
try:
    train_sample = np.load(os.path.join(DATA_DIR, "train.npy"))
    SEQ_LEN = train_sample.shape[1]
    FEATURE_DIM = train_sample.shape[2]
    print(f"📊 Auto-detected: SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    SEQ_LEN = 168
    FEATURE_DIM = 14
    print(f"⚠️ Using defaults: SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")

# MATCH ORIGINAL TRAINING PARAMETERS
HIDDEN_DIM = 128  # Match your original train_recon_tf.py
NUM_LAYERS = 3    # Match your original train_recon_tf.py

# ---------------- Model Reconstruction (Matching Original) ----------------
print("🔨 Building models (matching original training)...")
embedder = Embedder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=NUM_LAYERS, dropout=0.2)

# Build model shapes
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

print(f"📊 Model Parameters:")
print(f"  Embedder: {sum(np.prod(v.shape) for v in embedder.trainable_variables):,}")
print(f"  Recovery: {sum(np.prod(v.shape) for v in recovery.trainable_variables):,}")

# ---------------- Checkpoint Loading ----------------
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

# Try multiple checkpoint locations
checkpoint_locations = [
    CKPT_DIR,  # Original directory
    "outputs/checkpoints/recon_enhanced",  # Fallback to enhanced
]

checkpoint_loaded = False
loaded_from = ""

for checkpoint_dir in checkpoint_locations:
    if os.path.exists(checkpoint_dir):
        temp_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
        if temp_manager.latest_checkpoint:
            try:
                ckpt.restore(temp_manager.latest_checkpoint).expect_partial()
                print(f"✅ Restored from checkpoint: {temp_manager.latest_checkpoint}")
                checkpoint_loaded = True
                loaded_from = checkpoint_dir
                break
            except Exception as e:
                print(f"⚠️ Failed to load from {checkpoint_dir}: {e}")
                continue

if not checkpoint_loaded:
    # Try loading individual weight files
    weight_locations = [
        ("outputs/checkpoints/recon", "embedder_final.h5", "recovery_final.h5"),
        ("outputs/checkpoints/recon_enhanced", "enhanced_embedder_final.h5", "enhanced_recovery_final.h5"),
        ("outputs/checkpoints/recon", "embedder_final.weights.h5", "recovery_final.weights.h5"),
    ]
    
    for checkpoint_dir, embedder_file, recovery_file in weight_locations:
        embedder_path = os.path.join(checkpoint_dir, embedder_file)
        recovery_path = os.path.join(checkpoint_dir, recovery_file)
        
        if os.path.exists(embedder_path) and os.path.exists(recovery_path):
            try:
                embedder.load_weights(embedder_path)
                recovery.load_weights(recovery_path)
                print(f"✅ Loaded individual weight files from {checkpoint_dir}")
                checkpoint_loaded = True
                loaded_from = checkpoint_dir
                break
            except Exception as e:
                print(f"⚠️ Failed to load weights from {checkpoint_dir}: {e}")
                continue

if not checkpoint_loaded:
    print("❌ No compatible checkpoints or weight files found!")
    print("💡 Please run train_recon_tf.py first to train the models")
    print("📁 Checked directories:")
    for checkpoint_dir in checkpoint_locations:
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            print(f"   {checkpoint_dir}: {files}")
    raise SystemExit("Checkpoint loading failed")

# ---------------- Save Final Weights ----------------
print("\n💾 Saving final weights...")

# Create enhanced directory if we loaded from enhanced
if "enhanced" in loaded_from:
    save_dir = "outputs/checkpoints/recon_enhanced"
    embedder_filename = "enhanced_embedder_final.weights.h5"
    recovery_filename = "enhanced_recovery_final.weights.h5"
else:
    save_dir = "outputs/checkpoints/recon"
    embedder_filename = "embedder_final.weights.h5"
    recovery_filename = "recovery_final.weights.h5"

os.makedirs(save_dir, exist_ok=True)

embedder_path = os.path.join(save_dir, embedder_filename)
recovery_path = os.path.join(save_dir, recovery_filename)

embedder.save_weights(embedder_path)
recovery.save_weights(recovery_path)

print("✅ Saved weights:")
print(f"   Embedder: {embedder_path}")
print(f"   Recovery: {recovery_path}")

# ---------------- Reconstruction Example Generation ----------------
print("\n🔍 Generating reconstruction examples...")

try:
    test = np.load(os.path.join(DATA_DIR, "test.npy"))
    print(f"📊 Test data shape: {test.shape}")
    
    # Use examples for analysis
    examples = test[:5]
    
    # Forward pass
    h_ex = embedder(examples, training=False)
    x_rec = recovery(h_ex, training=False).numpy()
    
    print(f"✅ Generated reconstructions: {x_rec.shape}")

except Exception as e:
    print(f"❌ Error loading test data: {e}")
    # Create dummy data for testing
    examples = np.random.normal(0, 1, (3, SEQ_LEN, FEATURE_DIM)).astype(np.float32)
    h_ex = embedder(examples, training=False)
    x_rec = recovery(h_ex, training=False).numpy()
    print("⚠️ Using dummy data for reconstruction test")

# ---------------- Inverse Scaling ----------------
print("\n🔄 Applying inverse scaling...")

# Find scalers file
scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
if not os.path.exists(scalers_path):
    scalers_path = os.path.join(DATA_DIR, "scaler.pkl")

if os.path.exists(scalers_path):
    with open(scalers_path, "rb") as f:
        scaler = pickle.load(f)
    
    n, T, D = x_rec.shape
    
    try:
        if isinstance(scaler, list) and len(scaler) == D:
            # per-feature scalers
            x_rec_inv = np.zeros_like(x_rec)
            x_orig_inv = np.zeros_like(examples)
            for j, sc in enumerate(scaler):
                orig_shape = x_rec[:, :, j].shape
                x_rec_inv[:, :, j] = sc.inverse_transform(x_rec[:, :, j].reshape(-1, 1)).reshape(orig_shape)
                x_orig_inv[:, :, j] = sc.inverse_transform(examples[:, :, j].reshape(-1, 1)).reshape(orig_shape)
            print("✅ Applied per-feature inverse scaling")
        else:
            # single scaler
            original_shape = x_rec.shape
            x_rec_inv = scaler.inverse_transform(x_rec.reshape(-1, D)).reshape(original_shape)
            x_orig_inv = scaler.inverse_transform(examples.reshape(-1, D)).reshape(original_shape)
            print("✅ Applied single scaler inverse scaling")
    except Exception as e:
        print(f"❌ Inverse scaling failed: {e}")
        x_rec_inv = x_rec  # Fallback to scaled data
        x_orig_inv = examples
else:
    print("⚠️ No scalers found, using scaled data")
    x_rec_inv = x_rec
    x_orig_inv = examples

# ---------------- Diagnostics ----------------
print("\n📊 Reconstruction Diagnostics")
print("=" * 50)

# Load feature names
feature_names = []
feature_names_path = os.path.join(DATA_DIR, "features.txt")
if os.path.exists(feature_names_path):
    with open(feature_names_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
else:
    feature_names = [f"feature_{i}" for i in range(FEATURE_DIM)]

# Calculate metrics
overall_mae = float(np.mean(np.abs(x_orig_inv - x_rec_inv)))
overall_mse = float(np.mean((x_orig_inv - x_rec_inv) ** 2))

print(f"📈 Overall Metrics:")
print(f"  MAE:  {overall_mae:.6f}")
print(f"  MSE:  {overall_mse:.6f}")
print(f"  RMSE: {np.sqrt(overall_mse):.6f}")

print(f"\n🔍 Feature-wise MAE (first sequence):")
for i, name in enumerate(feature_names):
    orig_feat = x_orig_inv[0, :, i]
    rec_feat = x_rec_inv[0, :, i]
    mae = np.mean(np.abs(orig_feat - rec_feat))
    
    if name in ['Open', 'High', 'Low', 'Close']:
        # Calculate relative error for price features
        avg_value = np.mean(np.abs(orig_feat))
        if avg_value > 0:
            rel_error = (mae / avg_value) * 100
            status = "✅" if rel_error < 5 else "⚠️" if rel_error < 15 else "❌"
            print(f"  {name:15}: {mae:.4f} ({rel_error:.1f}%) {status}")
        else:
            print(f"  {name:15}: {mae:.6f}")
    else:
        status = "✅" if mae < 0.1 else "⚠️" if mae < 0.3 else "❌"
        print(f"  {name:15}: {mae:.6f} {status}")

# ---------------- Save Reconstruction Results ----------------
print("\n💾 Saving reconstruction artifacts...")

# Save reconstruction results
np.save(os.path.join(save_dir, "recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(save_dir, "recon_examples_rec.npy"), x_rec_inv)

# Save scaled versions too
np.save(os.path.join(save_dir, "recon_examples_orig_scaled.npy"), examples)
np.save(os.path.join(save_dir, "recon_examples_rec_scaled.npy"), x_rec)

# Save metrics
metrics = {
    'overall_mae': overall_mae,
    'overall_mse': overall_mse,
    'overall_rmse': np.sqrt(overall_mse),
    'feature_mae': {name: float(np.mean(np.abs(x_orig_inv[0, :, i] - x_rec_inv[0, :, i]))) 
                   for i, name in enumerate(feature_names)},
    'model_architecture': {
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'feature_dim': FEATURE_DIM,
        'seq_len': SEQ_LEN
    }
}

with open(os.path.join(save_dir, "recon_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Reconstruction artifacts saved:")
print(f"   Original data: recon_examples_orig.npy")
print(f"   Reconstructed: recon_examples_rec.npy")
print(f"   Metrics: recon_metrics.json")

# ---------------- Quality Assessment ----------------
print("\n🎯 RECONSTRUCTION QUALITY ASSESSMENT")
print("=" * 50)

# Count features by quality
excellent = sum(1 for mae in metrics['feature_mae'].values() if mae < 0.1)
good = sum(1 for mae in metrics['feature_mae'].values() if 0.1 <= mae < 0.3)
poor = sum(1 for mae in metrics['feature_mae'].values() if mae >= 0.3)

print(f"📊 Feature Reconstruction Quality:")
print(f"  ✅ Excellent: {excellent}/{len(feature_names)}")
print(f"  ⚠️ Good/Fair: {good}/{len(feature_names)}")
print(f"  ❌ Poor:      {poor}/{len(feature_names)}")

# Key features assessment
key_features = ['Open', 'High', 'Low', 'Close']
print(f"\n🔑 Key Price Features Status:")
for feature in key_features:
    if feature in metrics['feature_mae']:
        mae = metrics['feature_mae'][feature]
        status = "✅ Good" if mae < 0.1 else "⚠️ Fair" if mae < 0.3 else "❌ Poor"
        print(f"  {feature}: {status} (MAE: {mae:.4f})")

# Overall recommendation
if poor == 0 and excellent >= len(feature_names) // 2:
    print(f"\n🎉 EXCELLENT: Reconstruction quality is good!")
    print("   Proceed with TimeGAN training.")
elif poor <= 2:
    print(f"\n✅ ACCEPTABLE: Reconstruction quality is acceptable.")
    print("   Proceed with TimeGAN training.")
else:
    print(f"\n⚠️ NEEDS IMPROVEMENT: Reconstruction quality is poor.")
    print("   Consider retraining with more epochs.")

print(f"\n📁 All artifacts saved to: {save_dir}")
print("✅ Reconstruction artifacts process completed!")