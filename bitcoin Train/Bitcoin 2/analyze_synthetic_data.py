# src/analyze_synth_graphs.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Paths ----------------
DATA_DIR = "data/processed/crypto"
SYN_DIR = "outputs/synth_fixed"
OUT_DIR = "outputs/graphs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load Data ----------------
def load_real_data():
    real = np.load(os.path.join(DATA_DIR, "test.npy"))
    print("Real data shape:", real.shape)
    return real

def load_synth_data():
    # pick the latest inverse-scaled synthetic file
    files = sorted([f for f in os.listdir(SYN_DIR) if f.endswith(".npy")], reverse=True)
    synth_path = os.path.join(SYN_DIR, files[0])
    synth = np.load(synth_path)
    print("Synthetic data shape:", synth.shape)
    return synth

# ---------------- Feature Names ----------------
feat_path = os.path.join(DATA_DIR, "features.txt")
if os.path.exists(feat_path):
    with open(feat_path, "r") as f:
        feat_names = [line.strip() for line in f.readlines()]
else:
    feat_names = [f"feature_{i}" for i in range(14)]

# ---------------- Plotting Functions ----------------
def plot_feature_distributions(real, synth, feat_names, n_samples=1000):
    """Plot KDE/histogram for each feature"""
    n_features = len(feat_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols*5, n_rows*4))
    
    for i, feat in enumerate(feat_names):
        plt.subplot(n_rows, n_cols, i+1)
        # Sample subset to reduce overplot
        real_flat = real[:n_samples, :, i].flatten()
        synth_flat = synth[:n_samples, :, i].flatten()
        sns.kdeplot(real_flat, label='Real', color='blue')
        sns.kdeplot(synth_flat, label='Synth', color='red')
        plt.title(feat)
        plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "feature_distributions.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved feature distribution plot to", out_path)

def plot_random_sequences(real, synth, feat_names, n_seq=5):
    """Plot random sequences for comparison"""
    seq_idx_real = np.random.choice(real.shape[0], n_seq, replace=False)
    seq_idx_synth = np.random.choice(synth.shape[0], n_seq, replace=False)
    
    plt.figure(figsize=(15, n_seq*3))
    
    for i in range(n_seq):
        plt.subplot(n_seq, 1, i+1)
        # Plot Close price by default if available
        if 'Close' in feat_names:
            idx = feat_names.index('Close')
        else:
            idx = 0
        plt.plot(real[seq_idx_real[i], :, idx], label='Real', color='blue')
        plt.plot(synth[seq_idx_synth[i], :, idx], label='Synth', color='red')
        plt.title(f"Sequence {i+1} - Feature: {feat_names[idx]}")
        plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "random_sequences.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved random sequences plot to", out_path)

def plot_volatility(real, synth, n_samples=1000):
    """Compare volatility distributions (std per sequence)"""
    real_vol = np.std(real[:n_samples], axis=1).mean(axis=1)
    synth_vol = np.std(synth[:n_samples], axis=1).mean(axis=1)
    
    plt.figure(figsize=(8,6))
    sns.kdeplot(real_vol, label='Real', color='blue')
    sns.kdeplot(synth_vol, label='Synth', color='red')
    plt.title("Volatility per Sequence")
    plt.xlabel("Std")
    plt.legend()
    out_path = os.path.join(OUT_DIR, "volatility_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved volatility comparison plot to", out_path)

# ---------------- Main ----------------
if __name__ == "__main__":
    # real_data = load_real_data()
    real_train = np.load("data/processed/crypto/train.npy")
    real_test  = np.load("data/processed/crypto/test.npy")
    real_data  = np.concatenate([real_train, real_test], axis=0)
    print("Combined real data shape:", real_data.shape)

    synth_data = load_synth_data()
    synth_sample = synth_data[:2000]
    plot_feature_distributions(real_data, synth_sample, feat_names)
    plot_random_sequences(real_data, synth_sample, feat_names)
    plot_volatility(real_data, synth_sample)
    
    print("✅ Graph analysis completed.")


#     real_data = load_real_data()
#     synth_data = load_synth_data()
#     synth_sample = synth_data[:500]  # instead of using all 50,000
    
#     plot_feature_distributions(real_data, synth_sample, feat_names)
#     plot_random_sequences(real_data, synth_sample, feat_names)
#     plot_volatility(real_data, synth_sample)
    
#     print("✅ Graph analysis completed.")