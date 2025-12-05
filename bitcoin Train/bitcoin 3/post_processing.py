# post_processing.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

class SyntheticDataValidator:
    def __init__(self, real_data, synthetic_data):
        self.real = real_data
        self.synth = synthetic_data
        
    def compute_all_metrics(self):
        """Compute comprehensive quality metrics"""
        metrics = {}
        
        # 1. Basic statistics
        metrics['mean_similarity'] = self.mean_similarity()
        metrics['std_similarity'] = self.std_similarity()
        metrics['distribution_similarity'] = self.distribution_similarity()
        
        # 2. Temporal dynamics
        metrics['autocorrelation_similarity'] = self.autocorrelation_similarity()
        metrics['volatility_clustering'] = self.volatility_clustering_similarity()
        metrics['hurst_exponent_similarity'] = self.hurst_exponent_similarity()
        
        # 3. Dimensional metrics
        metrics['pca_similarity'] = self.pca_similarity()
        metrics['tsne_similarity'] = self.tsne_similarity()
        
        # 4. Predictive utility
        metrics['discriminative_score'] = self.discriminative_score()
        
        # Overall score (weighted average)
        weights = {
            'mean_similarity': 0.15,
            'std_similarity': 0.15,
            'distribution_similarity': 0.20,
            'autocorrelation_similarity': 0.15,
            'volatility_clustering': 0.10,
            'pca_similarity': 0.10,
            'discriminative_score': 0.15
        }
        
        overall_score = sum(metrics[k] * weights[k] for k in weights.keys())
        metrics['overall_quality_score'] = overall_score
        
        return metrics
    
    def mean_similarity(self):
        """Compare feature means"""
        real_mean = np.mean(self.real, axis=(0, 1))
        synth_mean = np.mean(self.synth, axis=(0, 1))
        
        # Normalized difference
        diff = np.abs(real_mean - synth_mean)
        std_real = np.std(self.real, axis=(0, 1))
        similarity = np.exp(-diff / (std_real + 1e-8))
        return np.mean(similarity)
    
    def distribution_similarity(self):
        """Compare distributions using KS test and Wasserstein distance"""
        from scipy.stats import ks_2samp
        from scipy.stats import wasserstein_distance
        
        scores = []
        n_features = self.real.shape[2]
        
        for i in range(n_features):
            real_flat = self.real[:, :, i].flatten()
            synth_flat = self.synth[:, :, i].flatten()
            
            # Sample for efficiency
            if len(real_flat) > 10000:
                idx = np.random.choice(len(real_flat), 10000, replace=False)
                real_flat = real_flat[idx]
                synth_flat = synth_flat[np.random.choice(len(synth_flat), 10000, replace=False)]
            
            # KS test
            ks_stat, _ = ks_2samp(real_flat, synth_flat)
            
            # Wasserstein distance
            w_dist = wasserstein_distance(real_flat, synth_flat)
            
            # Combine into score
            ks_score = 1 - ks_stat
            w_score = np.exp(-w_dist)
            scores.append((ks_score + w_score) / 2)
        
        return np.mean(scores)
    
    def autocorrelation_similarity(self, max_lag=20):
        """Compare autocorrelation structure"""
        def compute_acf(series, max_lag):
            acfs = []
            for lag in range(1, max_lag+1):
                if len(series) > lag:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    acfs.append(np.abs(corr))
            return np.array(acfs)
        
        scores = []
        for feat in range(min(5, self.real.shape[2])):  # Check first 5 features
            real_acf = compute_acf(self.real[:, :, feat].flatten(), max_lag)
            synth_acf = compute_acf(self.synth[:, :, feat].flatten(), max_lag)
            
            if len(real_acf) > 0 and len(synth_acf) > 0:
                min_len = min(len(real_acf), len(synth_acf))
                diff = np.mean(np.abs(real_acf[:min_len] - synth_acf[:min_len]))
                scores.append(np.exp(-diff))
        
        return np.mean(scores) if scores else 0.0
    
    def volatility_clustering_similarity(self):
        """Compare volatility clustering (ARCH effects)"""
        def compute_vol_clustering(returns):
            abs_returns = np.abs(returns)
            lagged = abs_returns[:-1] * abs_returns[1:]
            return np.mean(lagged) / (np.mean(abs_returns) ** 2 + 1e-8)
        
        # Use price returns (assuming first feature is price)
        real_returns = self.real[:, 1:, 0] - self.real[:, :-1, 0]
        synth_returns = self.synth[:, 1:, 0] - self.synth[:, :-1, 0]
        
        real_vc = compute_vol_clustering(real_returns.flatten())
        synth_vc = compute_vol_clustering(synth_returns.flatten())
        
        diff = np.abs(real_vc - synth_vc)
        return np.exp(-diff)
    
    def hurst_exponent_similarity(self):
        """Compare long-term memory using Hurst exponent"""
        from hurst import compute_Hc
        
        def estimate_hurst(series):
            try:
                H, _, _ = compute_Hc(series, kind='price', simplified=True)
                return H
            except:
                return 0.5
        
        real_h = estimate_hurst(self.real[:, :, 0].flatten())
        synth_h = estimate_hurst(self.synth[:, :, 0].flatten())
        
        diff = np.abs(real_h - synth_h)
        return np.exp(-diff * 10)  # Scale since H ∈ [0, 1]
    
    def pca_similarity(self):
        """Compare PCA reconstruction error"""
        from sklearn.decomposition import PCA
        
        # Flatten sequences
        real_flat = self.real.reshape(-1, self.real.shape[2])
        synth_flat = self.synth.reshape(-1, self.synth.shape[2])
        
        # Fit PCA on real data
        pca = PCA(n_components=min(10, real_flat.shape[1]))
        pca.fit(real_flat)
        
        # Transform both datasets
        real_transformed = pca.transform(real_flat)
        synth_transformed = pca.transform(synth_flat)
        
        # Compare distributions in PCA space
        real_mean = np.mean(real_transformed, axis=0)
        synth_mean = np.mean(synth_transformed, axis=0)
        
        real_cov = np.cov(real_transformed.T)
        synth_cov = np.cov(synth_transformed.T)
        
        # Bhattacharyya distance
        mean_diff = real_mean - synth_mean
        avg_cov = (real_cov + synth_cov) / 2
        try:
            # Mahalanobis-like distance
            dist = 0.125 * mean_diff @ np.linalg.inv(avg_cov) @ mean_diff.T
            + 0.5 * np.log(np.linalg.det(avg_cov) / 
                          np.sqrt(np.linalg.det(real_cov) * np.linalg.det(synth_cov)))
            return np.exp(-dist)
        except:
            return 0.5
    
    def discriminative_score(self):
        """Train a classifier to distinguish real vs synthetic"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        # Prepare data
        X_real = self.real.reshape(self.real.shape[0], -1)
        X_synth = self.synth.reshape(self.synth.shape[0], -1)
        
        X = np.vstack([X_real, X_synth])
        y = np.hstack([np.zeros(len(X_real)), np.ones(len(X_synth))])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X, y = X[shuffle_idx], y[shuffle_idx]
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            # Perfect discrimination = 1.0, random = 0.5
            # We want close to 0.5 (hard to distinguish)
            discriminative_score = 1 - 2 * np.abs(np.mean(scores) - 0.5)
            return max(0, discriminative_score)
        except:
            return 0.5

def apply_financial_constraints(synthetic_data, feature_names):
    """Apply domain-specific financial constraints"""
    data = np.copy(synthetic_data)
    
    # Map feature indices
    idx_map = {name: i for i, name in enumerate(feature_names)}
    
    # Apply OHLC constraints
    if all(k in idx_map for k in ['Open', 'High', 'Low', 'Close']):
        o, h, l, c = idx_map['Open'], idx_map['High'], idx_map['Low'], idx_map['Close']
        
        for i in range(data.shape[0]):
            for t in range(data.shape[1]):
                # Ensure High >= max(Open, Close)
                current_high = max(data[i, t, o], data[i, t, c])
                if data[i, t, h] < current_high:
                    data[i, t, h] = current_high * (1 + np.random.uniform(0, 0.01))
                
                # Ensure Low <= min(Open, Close)
                current_low = min(data[i, t, o], data[i, t, c])
                if data[i, t, l] > current_low:
                    data[i, t, l] = current_low * (1 - np.random.uniform(0, 0.01))
                
                # Ensure positive prices
                for price_idx in [o, h, l, c]:
                    data[i, t, price_idx] = max(data[i, t, price_idx], 0.01)
    
    # Volume constraints
    if 'Volume' in idx_map:
        v = idx_map['Volume']
        data[:, :, v] = np.maximum(data[:, :, v], 0)
    
    # Returns should be reasonably bounded
    if 'Returns' in idx_map:
        r = idx_map['Returns']
        # Clip extreme returns (±50%)
        data[:, :, r] = np.clip(data[:, :, r], -0.5, 0.5)
    
    # Volatility should be positive
    if 'Volatility' in idx_map:
        vol = idx_map['Volatility']
        data[:, :, vol] = np.maximum(data[:, :, vol], 0)
    
    # RSI should be between 0-100
    if 'RSI' in idx_map:
        rsi = idx_map['RSI']
        data[:, :, rsi] = np.clip(data[:, :, rsi], 0, 100)
    
    return data

def generate_high_quality_synthetic(n_samples=50000):
    """Complete generation pipeline with quality assurance"""
    
    # Load trained models
    generator = EnhancedGenerator(z_dim=64, hidden_dim=128, num_layers=3)
    supervisor = EnhancedSupervisor(hidden_dim=128, num_layers=2)
    recovery = EnhancedRecovery(hidden_dim=128, output_dim=14, num_layers=2)
    
    # Load weights
    generator.load_weights('outputs/checkpoints/timegan_enhanced/best_generator.h5')
    supervisor.load_weights('outputs/checkpoints/timegan_enhanced/best_supervisor.h5')
    recovery.load_weights('outputs/checkpoints/timegan_enhanced/best_recovery.h5')
    
    # Generate in batches
    batch_size = 256
    synthetic_batches = []
    
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        
        # Sample noise
        z = tf.random.normal([current_batch, 168, 64], stddev=0.5)
        
        # Generate
        e_hat = generator(z, training=False)
        h_hat = supervisor(e_hat, training=False)
        x_hat = recovery(h_hat, training=False).numpy()
        
        # Clean NaN/Inf
        x_hat = np.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        synthetic_batches.append(x_hat)
    
    synthetic = np.concatenate(synthetic_batches, axis=0)
    
    # Load feature names
    with open('data/processed/crypto/features.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Apply constraints
    synthetic_constrained = apply_financial_constraints(synthetic, feature_names)
    
    # Load real data for validation
    real_data = []
    for split in ['train', 'val', 'test']:
        data = np.load(f'data/processed/crypto/{split}.npy')
        real_data.append(data)
    real_data = np.concatenate(real_data, axis=0)
    
    # Validate quality
    validator = SyntheticDataValidator(real_data[:10000], synthetic_constrained[:10000])
    metrics = validator.compute_all_metrics()
    
    print("\n=== SYNTHETIC DATA QUALITY METRICS ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save if quality is high enough
    if metrics['overall_quality_score'] > 0.85:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'outputs/synthetic/high_quality_{timestamp}.npy'
        np.save(save_path, synthetic_constrained)
        
        # Also save metrics
        metrics_path = f'outputs/synthetic/metrics_{timestamp}.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        
        print(f"\n✅ High-quality synthetic data saved to {save_path}")
        print(f"📊 Metrics saved to {metrics_path}")
        
        return synthetic_constrained, metrics
    else:
        print(f"\n❌ Quality score {metrics['overall_quality_score']:.4f} below threshold (0.85)")
        return None, metrics

if __name__ == "__main__":
    synthetic_data, metrics = generate_high_quality_synthetic(50000)