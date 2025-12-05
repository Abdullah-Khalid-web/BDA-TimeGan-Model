import os
import pickle
import numpy as np
import tensorflow as tf
from timegan_tf import Generator, Supervisor, Recovery
import datetime

# ----------------- Configuration -----------------
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_optimized"
OUT_DIR = "outputs/synth_fixed"
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
HIDDEN_DIM = 64
Z_DIM = 32

print(f"🎯 Fixed Generation Configuration:")
print(f"   Sequence Length: {SEQ_LEN}")
print(f"   Feature Dimension: {FEATURE_DIM}")

# ----------------- Build Models -----------------
print("Building models...")
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=1)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=2)

# Build models
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ----------------- Load Model -----------------
def load_best_model():
    ckpt = tf.train.Checkpoint(generator=generator, supervisor=supervisor, recovery=recovery)
    manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=3)
    
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"✅ Loaded checkpoint: {manager.latest_checkpoint}")
        return True
    
    weight_files = {
        'generator': os.path.join(CKPT_DIR, "generator_final.weights.h5"),
        'supervisor': os.path.join(CKPT_DIR, "supervisor_final.weights.h5"), 
        'recovery': os.path.join(CKPT_DIR, "recovery_final.weights.h5")
    }
    
    if all(os.path.exists(f) for f in weight_files.values()):
        generator.load_weights(weight_files['generator'])
        supervisor.load_weights(weight_files['supervisor'])
        recovery.load_weights(weight_files['recovery'])
        print("✅ Loaded individual weights")
        return True
    
    return False

if not load_best_model():
    raise SystemExit("❌ No trained model found!")

# ----------------- Load Data -----------------
def load_real_data():
    all_data = []
    for split in ['train', 'val', 'test']:
        split_files = [f for f in os.listdir(DATA_DIR) if f.startswith(split) and f.endswith('.npy')]
        for file in split_files:
            data = np.load(os.path.join(DATA_DIR, file))
            all_data.append(data)
    
    if all_data:
        real_data = np.concatenate(all_data, axis=0)
        print(f"✅ Loaded {real_data.shape[0]} real sequences")
        return real_data
    return None

real_data = load_real_data()

# Feature names
feat_names_path = os.path.join(DATA_DIR, "features.txt")
if os.path.exists(feat_names_path):
    with open(feat_names_path, "r") as f:
        feat_names = [line.strip() for line in f.readlines()]
else:
    feat_names = [f"feature_{i}" for i in range(FEATURE_DIM)]

# ----------------- Generate Synthetic Data -----------------
def generate_synthetic(n_samples, batch_size=128):
    synthetic_batches = []
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        Z = tf.random.normal([current_batch, SEQ_LEN, Z_DIM], stddev=0.5)
        E_hat = generator(Z, training=False)
        H_hat = supervisor(E_hat, training=False)
        X_hat = recovery(H_hat, training=False).numpy()
        X_hat = np.nan_to_num(X_hat, nan=0.0, posinf=1.0, neginf=-1.0)
        synthetic_batches.append(X_hat)
    synthetic = np.concatenate(synthetic_batches, axis=0)
    print(f"✅ Generated synthetic data: {synthetic.shape}")
    return synthetic

# ----------------- Apply Inverse Scaling -----------------
def inverse_scaling(data):
    scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
    if not os.path.exists(scalers_path):
        scalers_path = os.path.join(DATA_DIR, "scaler.pkl")
    
    if os.path.exists(scalers_path):
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        X_inv = np.empty_like(data)
        try:
            if isinstance(scalers, list) and len(scalers) == data.shape[2]:
                for d in range(data.shape[2]):
                    flat = data[:, :, d].reshape(-1, 1)
                    X_inv[:, :, d] = scalers[d].inverse_transform(flat).reshape(data[:, :, d].shape)
            else:
                flat = data.reshape(-1, data.shape[2])
                X_inv = scalers.inverse_transform(flat).reshape(data.shape)
            print("✅ Inverse scaling applied")
            return X_inv
        except Exception as e:
            print(f"❌ Scaling failed: {e}")
            return data
    else:
        print("⚠️ No scalers found - returning scaled data")
        return data

# ----------------- Apply Constraints -----------------
def apply_constraints(data, feature_names):
    data_c = np.copy(data)
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}
    
    for f in ['Open', 'High', 'Low', 'Close']:
        if f not in feature_indices:
            continue
    open_idx, high_idx, low_idx, close_idx = [feature_indices[f] for f in ['Open','High','Low','Close']]
    
    for i in range(data_c.shape[0]):
        for t in range(data_c.shape[1]):
            o = data_c[i,t,open_idx]; h = data_c[i,t,high_idx]
            l = data_c[i,t,low_idx]; c = data_c[i,t,close_idx]
            max_price = max(o,c); min_price = min(o,c)
            if h < max_price: data_c[i,t,high_idx] = max_price + abs(o-c)*0.01
            if l > min_price: data_c[i,t,low_idx] = min_price - abs(o-c)*0.01
    # Ensure positive prices
    for f in ['Open','High','Low','Close']:
        if f in feature_indices:
            idx = feature_indices[f]
            data_c[:,:,idx] = np.maximum(data_c[:,:,idx], 0.01)
    # Volume non-negative
    if 'Volume' in feature_indices:
        idx = feature_indices['Volume']
        data_c[:,:,idx] = np.maximum(data_c[:,:,idx], 0.0)
    
    return data_c

# ----------------- Quality Assessment -----------------
def quality_score(synthetic, real, feature_names):
    if real is None:
        return 0.0, []
    real_flat = real.reshape(-1, real.shape[2])
    synth_flat = synthetic.reshape(-1, synthetic.shape[2])
    results = []
    scores = []
    for i,name in enumerate(feature_names):
        r,s = real_flat[:,i], synth_flat[:,i]
        # Remove extremes
        q1,q99 = np.percentile(r,[1,99])
        r_clean = r[(r>=q1)&(r<=q99)]
        q1,q99 = np.percentile(s,[1,99])
        s_clean = s[(s>=q1)&(s<=q99)]
        if len(r_clean)==0 or len(s_clean)==0: continue
        r_mean,r_std = r_clean.mean(), r_clean.std()
        s_mean,s_std = s_clean.mean(), s_clean.std()
        mean_diff = abs(s_mean-r_mean)
        std_diff = abs(s_std-r_std)
        # Simple thresholds
        if mean_diff<0.5 and std_diff<0.5: score = 1.0; status='EXCELLENT'
        elif mean_diff<1.0 and std_diff<1.0: score=0.8; status='GOOD'
        elif mean_diff<2.0 and std_diff<2.0: score=0.5; status='FAIR'
        else: score=0.2; status='POOR'
        scores.append(score)
        results.append({'feature':name,'real_mean':r_mean,'synth_mean':s_mean,
                        'real_std':r_std,'synth_std':s_std,'status':status})
    overall = np.mean(scores)*100 if scores else 0.0
    return overall, results

# ----------------- Save Outputs -----------------
def save_outputs(synthetic_scaled, synthetic_inv, feature_names):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save npy
    scaled_path = os.path.join(OUT_DIR, f"synthetic_scaled_{timestamp}.npy")
    inv_path = os.path.join(OUT_DIR, f"synthetic_inverse_{timestamp}.npy")
    np.save(scaled_path, synthetic_scaled)
    np.save(inv_path, synthetic_inv)
    # Save CSV sample (first 1000 sequences)
    csv_path = os.path.join(OUT_DIR, f"sample_scaled_{timestamp}.csv")
    sample_flat = synthetic_scaled[:1000].reshape(1000,-1)
    column_names = []
    for t in range(SEQ_LEN):
        for f in feature_names:
            column_names.append(f"{f}_t{t+1}")
    np.savetxt(csv_path, sample_flat, delimiter=",", header=",".join(column_names), comments='')
    return scaled_path, inv_path, csv_path

# ----------------- Main -----------------
if __name__ == "__main__":
    N_SAMPLES = 50000
    print(f"🎯 Generating {N_SAMPLES} synthetic sequences...")
    synthetic_scaled = generate_synthetic(N_SAMPLES, batch_size=256)
    synthetic_scaled = apply_constraints(synthetic_scaled, feat_names)
    synthetic_inv = inverse_scaling(synthetic_scaled)
    score, validation = quality_score(synthetic_scaled, real_data, feat_names)
    scaled_path, inv_path, csv_path = save_outputs(synthetic_scaled, synthetic_inv, feat_names)
    print(f"📊 Overall quality score: {score:.1f}%")
    print(f"✅ Scaled saved: {scaled_path}")
    print(f"✅ Inverse saved: {inv_path}")
    print(f"✅ CSV sample saved: {csv_path}")
