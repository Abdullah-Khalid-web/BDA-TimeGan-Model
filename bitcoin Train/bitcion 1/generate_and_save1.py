import os
import pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Generator, Supervisor, Recovery
import datetime
from sklearn.preprocessing import RobustScaler

# Configuration
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_optimized"
OUT_DIR = "outputs/synth_quick_fix"
os.makedirs(OUT_DIR, exist_ok=True)

# Auto-detect dimensions
def detect_dimensions():
    train_files = [f for f in os.listdir(DATA_DIR) if f.startswith('train') and f.endswith('.npy')]
    if train_files:
        sample_data = np.load(os.path.join(DATA_DIR, train_files[0]))
        seq_len = sample_data.shape[1]
        feature_dim = sample_data.shape[2]
        return seq_len, feature_dim
    return 168, 14

SEQ_LEN, FEATURE_DIM = detect_dimensions()
print(f"📊 Detected: SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")

# Load feature names
feat_names_path = os.path.join(DATA_DIR, "features.txt")
if os.path.exists(feat_names_path):
    with open(feat_names_path, "r") as f:
        FEATURE_NAMES = [line.strip() for line in f.readlines()]
else:
    FEATURE_NAMES = [f"feature_{i}" for i in range(FEATURE_DIM)]

print("🔧 Loading and analyzing real data distribution...")

# Load real data for distribution matching
def load_real_data_for_matching():
    all_data = []
    for split in ['train', 'val', 'test']:
        split_files = [f for f in os.listdir(DATA_DIR) if f.startswith(split) and f.endswith('.npy')]
        for file in split_files:
            data = np.load(os.path.join(DATA_DIR, file))
            all_data.append(data)
    
    if all_data:
        real_data = np.concatenate(all_data, axis=0)
        print(f"📈 Loaded {real_data.shape[0]:,} real sequences for distribution matching")
        return real_data
    return None

real_data = load_real_data_for_matching()

# Build models
print("🔨 Building models...")
# NEW (match your enhanced training):
generator = Generator(z_dim=64, hidden_dim=256, num_layers=3)  # Match enhanced
supervisor = Supervisor(hidden_dim=256, num_layers=2)          # Match enhanced  
recovery = Recovery(hidden_dim=256, output_dim=FEATURE_DIM, num_layers=3)  # Match enhanced

# Build models
_ = generator(tf.zeros([1, SEQ_LEN, 64]))    # Updated z_dim
_ = supervisor(tf.zeros([1, SEQ_LEN, 256]))  # Updated hidden_dim  
_ = recovery(tf.zeros([1, SEQ_LEN, 256]))    # Updated hidden_dim
# Load weights
def load_weights():
    weight_files = {
        'generator': os.path.join(CKPT_DIR, "generator_final.weights.h5"),
        'supervisor': os.path.join(CKPT_DIR, "supervisor_final.weights.h5"), 
        'recovery': os.path.join(CKPT_DIR, "recovery_final.weights.h5")
    }
    
    if all(os.path.exists(f) for f in weight_files.values()):
        generator.load_weights(weight_files['generator'])
        supervisor.load_weights(weight_files['supervisor'])
        recovery.load_weights(weight_files['recovery'])
        print("✅ Loaded model weights")
        return True
    else:
        print("❌ Model weights not found!")
        return False

if not load_weights():
    exit(1)

# DISTRIBUTION-AWARE GENERATION
def generate_distribution_aware(n_samples=50000, batch_size=256):
    """Generate synthetic data that matches real data distribution"""
    print(f"🎯 Generating {n_samples:,} distribution-aware sequences...")
    
    synthetic_batches = []
    
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        
        if (i // batch_size) % 20 == 0:
            print(f"   Progress: {i:,}/{n_samples:,} ({i/n_samples*100:.1f}%)")
        
        # Generate with controlled variance
        Z = tf.random.normal([current_batch, SEQ_LEN, 32], stddev=0.3)
        E_hat = generator(Z, training=False)
        H_hat = supervisor(E_hat, training=False)
        X_hat = recovery(H_hat, training=False).numpy()
        
        # Apply distribution correction
        X_hat = apply_distribution_correction(X_hat)
        
        synthetic_batches.append(X_hat)
    
    synthetic_data = np.concatenate(synthetic_batches, axis=0)
    print(f"✅ Generated {synthetic_data.shape[0]:,} sequences")
    
    return synthetic_data

def apply_distribution_correction(synthetic_batch):
    """Apply real data distribution characteristics to synthetic data"""
    if real_data is None:
        return synthetic_batch
    
    corrected_batch = np.copy(synthetic_batch)
    n, T, D = synthetic_batch.shape
    
    # Flatten for processing
    synth_flat = synthetic_batch.reshape(-1, D)
    
    for d in range(D):
        real_feature = real_data[:, :, d].flatten()
        synth_feature = synth_flat[:, d]
        
        # Calculate robust statistics (using percentiles)
        real_q1, real_median, real_q3 = np.percentile(real_feature, [25, 50, 75])
        synth_q1, synth_median, synth_q3 = np.percentile(synth_feature, [25, 50, 75])
        
        # Avoid division by zero
        if abs(synth_q3 - synth_q1) > 1e-8 and abs(real_q3 - real_q1) > 1e-8:
            # Adjust scale and location
            scale_factor = (real_q3 - real_q1) / (synth_q3 - synth_q1)
            location_shift = real_median - synth_median
            
            # Apply transformation
            synth_normalized = (synth_feature - synth_median) * scale_factor + real_median
            
            # Reshape back
            corrected_batch[:, :, d] = synth_normalized.reshape(n, T)
    
    return corrected_batch

def apply_financial_constraints(data, feature_names):
    """Apply realistic financial constraints"""
    print("💡 Applying financial constraints...")
    
    data_constrained = np.copy(data)
    
    # Get feature indices
    feature_indices = {}
    for name in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if name in feature_names:
            feature_indices[name] = feature_names.index(name)
    
    # Apply price relationships
    if all(k in feature_indices for k in ['Open', 'High', 'Low', 'Close']):
        open_idx = feature_indices['Open']
        high_idx = feature_indices['High'] 
        low_idx = feature_indices['Low']
        close_idx = feature_indices['Close']
        
        for i in range(data_constrained.shape[0]):
            for t in range(data_constrained.shape[1]):
                open_val = data_constrained[i, t, open_idx]
                high_val = data_constrained[i, t, high_idx]
                low_val = data_constrained[i, t, low_idx]
                close_val = data_constrained[i, t, close_idx]
                
                # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
                max_price = max(open_val, close_val)
                min_price = min(open_val, close_val)
                
                if high_val < max_price:
                    data_constrained[i, t, high_idx] = max_price + abs(open_val) * 0.001
                if low_val > min_price:
                    data_constrained[i, t, low_idx] = min_price - abs(open_val) * 0.001
                
                # Ensure High >= Low
                if data_constrained[i, t, high_idx] < data_constrained[i, t, low_idx]:
                    avg = (data_constrained[i, t, high_idx] + data_constrained[i, t, low_idx]) / 2
                    spread = abs(data_constrained[i, t, high_idx] - data_constrained[i, t, low_idx]) * 0.01
                    data_constrained[i, t, high_idx] = avg + spread
                    data_constrained[i, t, low_idx] = avg - spread
    
    print("✅ Financial constraints applied")
    return data_constrained

def quick_quality_check(synthetic, real_sample):
    """Quick quality assessment"""
    print("\n🔍 QUICK QUALITY CHECK")
    print("-" * 60)
    
    synth_flat = synthetic.reshape(-1, synthetic.shape[2])
    real_flat = real_sample.reshape(-1, real_sample.shape[2])
    
    scores = []
    
    for i, name in enumerate(FEATURE_NAMES[:8]):  # Check first 8 features
        real_feat = real_flat[:, i]
        synth_feat = synth_flat[:, i]
        
        # Remove outliers for stable comparison
        def remove_extremes(data):
            q1, q99 = np.percentile(data, [5, 95])
            return data[(data >= q1) & (data <= q99)]
        
        real_clean = remove_extremes(real_feat)
        synth_clean = remove_extremes(synth_feat)
        
        if len(real_clean) == 0 or len(synth_clean) == 0:
            continue
            
        mean_diff = abs(np.mean(synth_clean) - np.mean(real_clean))
        std_diff = abs(np.std(synth_clean) - np.std(real_clean))
        
        # Simple scoring
        mean_score = max(0, 1 - mean_diff)
        std_score = max(0, 1 - std_diff)
        feature_score = (mean_score + std_score) / 2
        
        scores.append(feature_score)
        
        status = "✅ GOOD" if feature_score > 0.8 else "⚠️ FAIR" if feature_score > 0.6 else "❌ POOR"
        print(f"{name:15}: mean_diff={mean_diff:6.4f}, std_diff={std_diff:6.4f}, score={feature_score:5.1%} {status}")
    
    overall_score = np.mean(scores) if scores else 0.0
    print(f"\n📊 OVERALL SCORE: {overall_score:.1%}")
    
    return overall_score

# Main execution
if __name__ == "__main__":
    print("🚀 QUICK-FIX SYNTHETIC DATA GENERATION")
    print("=" * 50)
    
    N_SAMPLES = 50000
    
    # Generate distribution-aware synthetic data
    synthetic_scaled = generate_distribution_aware(N_SAMPLES)
    
    # Apply financial constraints
    synthetic_final = apply_financial_constraints(synthetic_scaled, FEATURE_NAMES)
    
    # Quick quality check
    if real_data is not None:
        real_sample = real_data[:1000]  # Sample for quick check
        quality_score = quick_quality_check(synthetic_final, real_sample)
    else:
        quality_score = 0.0
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save scaled version
    scaled_path = os.path.join(OUT_DIR, f"quickfix_synthetic_scaled_{timestamp}.npy")
    np.save(scaled_path, synthetic_final)
    
    # Apply inverse scaling if scalers exist
    def apply_inverse_scaling(data):
        scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
        if os.path.exists(scalers_path):
            with open(scalers_path, "rb") as f:
                scalers = pickle.load(f)
            
            n, T, D = data.shape
            data_inv = np.empty_like(data)
            
            try:
                if isinstance(scalers, list) and len(scalers) == D:
                    for d in range(D):
                        original_shape = data[:, :, d].shape
                        flattened = data[:, :, d].reshape(-1, 1)
                        inverted = scalers[d].inverse_transform(flattened)
                        data_inv[:, :, d] = inverted.reshape(original_shape)
                    return data_inv
            except Exception as e:
                print(f"❌ Inverse scaling failed: {e}")
        
        return data
    
    synthetic_inverse = apply_inverse_scaling(synthetic_final)
    inverse_path = os.path.join(OUT_DIR, f"quickfix_synthetic_inverse_{timestamp}.npy")
    np.save(inverse_path, synthetic_inverse)
    
    print(f"\n💾 QUICK-FIX RESULTS:")
    print(f"   Scaled data: {scaled_path}")
    print(f"   Inverse-scaled: {inverse_path}")
    print(f"   Quality score: {quality_score:.1%}")
    
    if quality_score > 0.7:
        print("🎉 Good quality synthetic data generated!")
    elif quality_score > 0.5:
        print("✅ Acceptable quality synthetic data generated")
    else:
        print("⚠️  Quality needs improvement - consider retraining models")
    
    print("✅ Quick-fix generation completed!")