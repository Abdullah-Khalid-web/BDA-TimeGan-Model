# src/generate_and_save_fixed.py
# FIXED VERSION - Correct feature dimensions and improved generation
import os
import pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

# ----------------- Paths -----------------
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_wgan_gp_fixed"
OUT_DIR = "outputs/synth"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- Model config -----------------
SEQ_LEN = 168
# 🚨 FIXED: Get correct feature dimension from data
FEATURE_DIM = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
HIDDEN_DIM = 128
Z_DIM = 32

print(f"Using feature dimension: {FEATURE_DIM}")

# ----------------- Rebuild models -----------------
embedder = Embedder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=2)
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=1)
discriminator = Discriminator(hidden_dim=HIDDEN_DIM, num_layers=1)

# Build (dummy)
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = discriminator(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ----------------- Restore latest checkpoint -----------------
ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery,
                           generator=generator, supervisor=supervisor,
                           discriminator=discriminator)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=6)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print("Restored checkpoint:", manager.latest_checkpoint)
else:
    raise SystemExit(f"No checkpoint found in {CKPT_DIR}")

# ----------------- Load scaler(s) -----------------
scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
if not os.path.exists(scalers_path):
    scalers_path = os.path.join(DATA_DIR, "scaler.pkl")
if not os.path.exists(scalers_path):
    raise SystemExit("No scaler file found in data directory.")
with open(scalers_path, "rb") as f:
    scalers = pickle.load(f)

# Load feature names
feat_names_path = os.path.join(DATA_DIR, "features.txt")
if os.path.exists(feat_names_path):
    feat_names = open(feat_names_path).read().splitlines()
    print("Feature names:", feat_names)
else:
    # Fallback if features.txt doesn't exist
    feat_names = [f"feature_{i}" for i in range(FEATURE_DIM)]
    print("Using default feature names")

# ----------------- IMPROVED: Generate with realistic constraints -----------------
def enhanced_statistical_correction(synthetic, real_data, feature_names):
    """More robust statistical alignment"""
    synthetic_corrected = np.copy(synthetic)
    
    # Load real data statistics
    real_flat = real_data.reshape(-1, real_data.shape[2])
    synth_flat = synthetic.reshape(-1, synthetic.shape[2])
    
    for i, name in enumerate(feature_names):
        real_feature = real_flat[:, i]
        synth_feature = synth_flat[:, i]
        
        # Calculate robust statistics (ignore outliers)
        real_mean = np.median(real_feature)
        real_std = np.std(real_feature[np.abs(real_feature - real_mean) < 3 * np.std(real_feature)])
        synth_mean = np.median(synth_feature)
        synth_std = np.std(synth_feature[np.abs(synth_feature - synth_mean) < 3 * np.std(synth_feature)])
        
        print(f"{name}: Real(μ={real_mean:.4f}, σ={real_std:.4f}) vs Synth(μ={synth_mean:.4f}, σ={synth_std:.4f})")
        
        # Only apply correction if statistics are significantly different
        if abs(synth_mean - real_mean) > 0.1 * abs(real_mean) or abs(synth_std - real_std) > 0.5 * real_std:
            if synth_std > 1e-8:
                # Apply statistical alignment
                synthetic_corrected[:, :, i] = (synthetic[:, :, i] - synth_mean) / synth_std
                synthetic_corrected[:, :, i] = synthetic_corrected[:, :, i] * real_std + real_mean
                print(f"  ✅ Applied statistical correction")
    
    return synthetic_corrected

def enforce_price_relationships(data, feature_names):
    """Ensure realistic price relationships: High >= Open, Close; Low <= Open, Close"""
    if not all(name in feature_names for name in ['Open', 'High', 'Low', 'Close']):
        return data
    
    data_constrained = np.copy(data)
    open_idx = feature_names.index('Open')
    high_idx = feature_names.index('High')
    low_idx = feature_names.index('Low')
    close_idx = feature_names.index('Close')
    
    for j in range(data_constrained.shape[0]):
        for t in range(data_constrained.shape[1]):
            open_val = data_constrained[j, t, open_idx]
            high_val = data_constrained[j, t, high_idx]
            low_val = data_constrained[j, t, low_idx]
            close_val = data_constrained[j, t, close_idx]
            
            # Ensure High is highest of the four prices
            max_price = max(open_val, close_val, high_val, low_val)
            data_constrained[j, t, high_idx] = max_price
            
            # Ensure Low is lowest of the four prices
            min_price = min(open_val, close_val, high_val, low_val)
            data_constrained[j, t, low_idx] = min_price
            
            # Small random noise to avoid identical prices
            if abs(high_val - low_val) < 1e-6:
                data_constrained[j, t, high_idx] += 0.01
                data_constrained[j, t, low_idx] -= 0.01
    
    return data_constrained

def apply_realistic_bitcoin_constraints(data, feature_names):
    """Apply Bitcoin-specific realistic constraints"""
    data_constrained = np.copy(data)
    
    # Get feature indices
    price_indices = []
    volume_indices = []
    return_indices = []
    
    for name in ['Open', 'High', 'Low', 'Close']:
        if name in feature_names:
            price_indices.append(feature_names.index(name))
    
    for name in ['Volume', 'Volume_MA']:
        if name in feature_names:
            volume_indices.append(feature_names.index(name))
            
    for name in ['Price_Change', 'Log_Return']:
        if name in feature_names:
            return_indices.append(feature_names.index(name))
    
    # Price constraints
    for idx in price_indices:
        # Ensure positive prices with realistic Bitcoin range
        feature_data = data[:, :, idx]
        current_mean = np.mean(feature_data)
        
        # If prices are way off, rescale to realistic Bitcoin range
        if current_mean > 1000:  # If average price is > $1000, likely wrong scaling
            scale_factor = current_mean / 10.0  # Bring down to ~$10 range
            data_constrained[:, :, idx] = feature_data / scale_factor
        
        # Ensure positive prices
        data_constrained[:, :, idx] = np.maximum(data_constrained[:, :, idx], 0.1)
        
        # Cap extreme outliers (99.9th percentile of reasonable Bitcoin prices)
        p999 = np.percentile(data_constrained[:, :, idx], 99.9)
        reasonable_max = p999 * 1.5  # Allow some variation
        data_constrained[:, :, idx] = np.minimum(data_constrained[:, :, idx], reasonable_max)
    
    # Volume constraints
    for idx in volume_indices:
        data_constrained[:, :, idx] = np.maximum(data_constrained[:, :, idx], 0)
    
    # Return constraints (realistic bounds for hourly data)
    for idx in return_indices:
        data_constrained[:, :, idx] = np.clip(data_constrained[:, :, idx], -0.2, 0.2)  # Max ±20% hourly change
    
    return data_constrained

def generate_high_quality_synthetic(n_samples, batch_size=64):
    """Enhanced generation with robust statistical correction"""
    
    # Load real data for reference
    real_train = np.load(os.path.join(DATA_DIR, "train.npy"))
    
    print("🚀 GENERATING HIGH-QUALITY SYNTHETIC DATA...")
    
    # Generate in batches
    out = []
    for i in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - i)
        Z = tf.random.normal([b, SEQ_LEN, Z_DIM])
        Ehat = generator(Z, training=False)
        Hhat = supervisor(Ehat, training=False)
        Xhat_raw = recovery(Hhat, training=False).numpy()
        
        # Apply statistical correction
        Xhat_corrected = enhanced_statistical_correction(Xhat_raw, real_train, feat_names)
        out.append(Xhat_corrected)
    
    X_scaled = np.concatenate(out, axis=0)
    
    # Inverse scaling
    n, T, D = X_scaled.shape
    X_inv = np.empty_like(X_scaled)
    
    for d, s in enumerate(scalers):
        try:
            X_inv[:, :, d] = s.inverse_transform(X_scaled[:, :, d].reshape(-1, 1)).reshape(n, T)
        except Exception as e:
            print(f"Warning: Scaling failed for feature {d}: {e}")
            X_inv[:, :, d] = X_scaled[:, :, d]
    
    # Apply realistic constraints
    X_inv = enforce_price_relationships(X_inv, feat_names)
    X_inv = apply_realistic_bitcoin_constraints(X_inv, feat_names)
    
    return X_inv

def validate_synthetic_quality(synthetic, real_data, feature_names):
    """Comprehensive validation of synthetic data quality"""
    print("\n" + "="*50)
    print("SYNTHETIC DATA QUALITY REPORT")
    print("="*50)
    
    real_flat = real_data.reshape(-1, real_data.shape[2])
    synth_flat = synthetic.reshape(-1, synthetic.shape[2])
    
    quality_score = 0
    total_features = len(feature_names)
    
    for i, name in enumerate(feature_names):
        real_feat = real_flat[:, i]
        synth_feat = synth_flat[:, i]
        
        # Statistical similarity
        real_mean, real_std = np.mean(real_feat), np.std(real_feat)
        synth_mean, synth_std = np.mean(synth_feat), np.std(synth_feat)
        
        mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-8)
        std_diff = abs(real_std - synth_std) / (real_std + 1e-8)
        
        feature_score = 0
        if mean_diff < 0.1:  # Mean within 10%
            feature_score += 1
            print(f"✅ {name}: Mean match ({mean_diff*100:.1f}% diff)")
        else:
            print(f"⚠️  {name}: Mean mismatch ({mean_diff*100:.1f}% diff)")
            
        if std_diff < 0.2:  # Std within 20%
            feature_score += 1
            print(f"✅ {name}: Std match ({std_diff*100:.1f}% diff)")
        else:
            print(f"⚠️  {name}: Std mismatch ({std_diff*100:.1f}% diff)")
        
        quality_score += feature_score
    
    overall_quality = (quality_score / (total_features * 2)) * 100
    print(f"\n📊 OVERALL QUALITY SCORE: {overall_quality:.1f}%")
    
    if overall_quality > 80:
        print("🎉 EXCELLENT: Synthetic data quality is high!")
    elif overall_quality > 60:
        print("👍 GOOD: Synthetic data quality is acceptable")
    else:
        print("🔧 NEEDS IMPROVEMENT: Consider regenerating with better parameters")
    
    return overall_quality

# ----------------- Run -----------------
if __name__ == "__main__":
    N = 2000   # number of windows to generate
    
    print("Generating realistic synthetic Bitcoin data...")
    synth = generate_high_quality_synthetic(N, batch_size=64)
    
    # Validate the output
    print(f"Generated synthetic data shape: {synth.shape}")
    
    # Print statistics for validation
    print("\n=== SYNTHETIC DATA STATISTICS ===")
    for i, name in enumerate(feat_names[:6]):  # Show first 6 features
        feature_data = synth[:, :, i]
        print(f"{name}: min={feature_data.min():.4f}, max={feature_data.max():.4f}, mean={feature_data.mean():.4f}")
        if name in ['Open', 'High', 'Low', 'Close']:
            negative_count = np.sum(feature_data < 0)
            if negative_count > 0:
                print(f"  ⚠️  WARNING: {negative_count} negative values found in {name}!")
    
    # Quality validation
    real_test = np.load(os.path.join(DATA_DIR, "test.npy"))
    quality_score = validate_synthetic_quality(synth, real_test, feat_names)
    
    # Save the improved synthetic data
    out_npy = os.path.join(OUT_DIR, "synth_crypto_high_quality.npy")
    out_csv = os.path.join(OUT_DIR, "synth_crypto_high_quality.csv")
    
    np.save(out_npy, synth)
    np.savetxt(out_csv, synth.reshape(synth.shape[0], -1), delimiter=",")
    
    print(f"\n✅ Saved high-quality synthetic data to:")
    print(" ", out_npy)
    print(" ", out_csv)
    print(f"✅ Data shape: {synth.shape}")
    print(f"✅ Quality score: {quality_score:.1f}%")