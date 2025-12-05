# Air Quality Synthetic Data Evaluation Script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import pickle
import os
import glob
import sys
from datetime import datetime

# ===========================================
# FIX: Force UTF-8 encoding for Windows
# ===========================================
def safe_print(message):
    """Safely print messages avoiding Unicode issues"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Replace Unicode characters with ASCII equivalents
        replacements = {
            '✅': '[SUCCESS]',
            '❌': '[FAILED]',
            '⚠️': '[WARNING]',
            '✓': '[OK]',
            '📊': '[PLOT]',
            '💾': '[SAVE]'
        }
        safe_message = message
        for uni_char, ascii_replacement in replacements.items():
            safe_message = safe_message.replace(uni_char, ascii_replacement)
        # Also handle unicode escape sequences
        safe_message = safe_message.replace('\u2705', '[SUCCESS]')
        safe_message = safe_message.replace('\u274c', '[FAILED]')
        safe_message = safe_message.replace('\u26a0', '[WARNING]')
        print(safe_message)

class SimpleAirQualityEvaluator:
    def __init__(self):
        """Initialize evaluator"""
        self.real_data = None
        self.synthetic_data = None
        
    def load_data(self):
        """Load real and synthetic data"""
        safe_print("Loading data...")
        
        # Load real data
        real_path = 'data/processed/air_quality/test.npy'
        if os.path.exists(real_path):
            self.real_data = np.load(real_path)
            safe_print("[SUCCESS] Real data loaded: {}".format(self.real_data.shape))
        else:
            # Try loading train data as fallback
            real_path = 'data/processed/air_quality/train.npy'
            if os.path.exists(real_path):
                self.real_data = np.load(real_path)[:1000]  # Use subset
                safe_print("[SUCCESS] Real data (train subset) loaded: {}".format(self.real_data.shape))
            else:
                safe_print("[FAILED] Real data not found!")
                return False
        
        # Load synthetic data (find latest)
        synth_dir = 'outputs/synthetic_air_quality'
        if not os.path.exists(synth_dir):
            safe_print("[FAILED] Synthetic data directory not found!")
            return False
        
        synth_files = glob.glob(os.path.join(synth_dir, 'synthetic_*.npy'))
        if not synth_files:
            safe_print("[FAILED] No synthetic files found!")
            return False
        
        # Use latest synthetic file
        latest_file = max(synth_files, key=os.path.getctime)
        self.synthetic_data = np.load(latest_file)
        safe_print("[SUCCESS] Synthetic data loaded: {} ({})".format(
            self.synthetic_data.shape, os.path.basename(latest_file)))
        
        # Ensure same number of samples for comparison
        n_samples = min(len(self.real_data), len(self.synthetic_data), 1000)
        self.real_data = self.real_data[:n_samples]
        self.synthetic_data = self.synthetic_data[:n_samples]
        
        safe_print("Using {} samples for evaluation".format(n_samples))
        return True
    
    def compute_basic_metrics(self):
        """Compute basic statistical metrics"""
        safe_print("\n" + "="*50)
        safe_print("BASIC STATISTICAL METRICS")
        safe_print("="*50)
        
        metrics = {}
        
        # Flatten data
        real_flat = self.real_data.reshape(-1, self.real_data.shape[2])
        synth_flat = self.synthetic_data.reshape(-1, self.synthetic_data.shape[2])
        
        # Sample for efficiency
        n_samples = min(5000, len(real_flat), len(synth_flat))
        real_sample = real_flat[np.random.choice(len(real_flat), n_samples, replace=False)]
        synth_sample = synth_flat[np.random.choice(len(synth_flat), n_samples, replace=False)]
        
        # Compare means
        real_mean = np.mean(real_sample, axis=0)
        synth_mean = np.mean(synth_sample, axis=0)
        
        try:
            metrics['mean_correlation'] = np.corrcoef(real_mean, synth_mean)[0, 1]
        except:
            metrics['mean_correlation'] = 0.0
        metrics['mean_mae'] = np.mean(np.abs(real_mean - synth_mean))
        
        # Compare standard deviations
        real_std = np.std(real_sample, axis=0)
        synth_std = np.std(synth_sample, axis=0)
        
        try:
            metrics['std_correlation'] = np.corrcoef(real_std, synth_std)[0, 1]
        except:
            metrics['std_correlation'] = 0.0
        metrics['std_mae'] = np.mean(np.abs(real_std - synth_std))
        
        # Compare min/max
        real_min = np.min(real_sample, axis=0)
        synth_min = np.min(synth_sample, axis=0)
        real_max = np.max(real_sample, axis=0)
        synth_max = np.max(synth_sample, axis=0)
        
        metrics['range_similarity'] = np.mean(
            1 - np.abs((real_max - real_min) - (synth_max - synth_min)) / (real_max - real_min + 1e-8)
        )
        
        # Print results
        safe_print("Mean Correlation: {:.4f}".format(metrics['mean_correlation']))
        safe_print("Std Correlation:  {:.4f}".format(metrics['std_correlation']))
        safe_print("Range Similarity: {:.4f}".format(metrics['range_similarity']))
        safe_print("Mean MAE:         {:.4f}".format(metrics['mean_mae']))
        safe_print("Std MAE:          {:.4f}".format(metrics['std_mae']))
        
        return metrics
    
    def compute_distribution_metrics(self):
        """Compute distribution similarity metrics"""
        safe_print("\n" + "="*50)
        safe_print("DISTRIBUTION SIMILARITY METRICS")
        safe_print("="*50)
        
        metrics = {}
        
        # Flatten data
        real_flat = self.real_data.reshape(-1, self.real_data.shape[2])
        synth_flat = self.synthetic_data.reshape(-1, self.synthetic_data.shape[2])
        
        # Sample for efficiency
        n_samples = min(5000, len(real_flat), len(synth_flat))
        real_sample = real_flat[np.random.choice(len(real_flat), n_samples, replace=False)]
        synth_sample = synth_flat[np.random.choice(len(synth_flat), n_samples, replace=False)]
        
        # Kolmogorov-Smirnov test for first few features
        ks_scores = []
        n_features_to_test = min(5, real_sample.shape[1])
        
        for i in range(n_features_to_test):
            try:
                stat, p_value = stats.ks_2samp(real_sample[:, i], synth_sample[:, i])
                ks_scores.append(p_value)
            except:
                ks_scores.append(0.0)
        
        metrics['ks_mean_pvalue'] = np.mean(ks_scores) if ks_scores else 0.0
        metrics['ks_median_pvalue'] = np.median(ks_scores) if ks_scores else 0.0
        
        # Wasserstein distance
        wasserstein_distances = []
        for i in range(n_features_to_test):
            try:
                wd = stats.wasserstein_distance(real_sample[:, i], synth_sample[:, i])
                wasserstein_distances.append(wd)
            except:
                wasserstein_distances.append(1.0)
        
        metrics['wasserstein_mean'] = np.mean(wasserstein_distances) if wasserstein_distances else 1.0
        
        # Print results
        safe_print("KS Test Mean p-value:    {:.4f}".format(metrics['ks_mean_pvalue']))
        safe_print("KS Test Median p-value:  {:.4f}".format(metrics['ks_median_pvalue']))
        safe_print("Wasserstein Mean Dist:   {:.4f}".format(metrics['wasserstein_mean']))
        
        safe_print("\nKS Test p-values by feature:")
        for i, pval in enumerate(ks_scores):
            safe_print("  Feature {}: {:.4f}".format(i, pval))
        
        return metrics
    
    def compute_temporal_metrics(self):
        """Compute temporal pattern metrics"""
        safe_print("\n" + "="*50)
        safe_print("TEMPORAL PATTERN METRICS")
        safe_print("="*50)
        
        metrics = {}
        
        # Auto-correlation comparison
        def compute_autocorrelation(data, max_lag=5):
            acs = []
            n_sequences = min(100, len(data))
            
            for seq_idx in range(n_sequences):
                for feat in range(min(3, data.shape[2])):  # First 3 features
                    series = data[seq_idx, :, feat]
                    for lag in range(1, min(max_lag + 1, len(series))):
                        if len(series) > lag:
                            try:
                                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                                if not np.isnan(corr):
                                    acs.append(np.abs(corr))
                            except:
                                continue
            
            return np.mean(acs) if acs else 0
        
        real_ac = compute_autocorrelation(self.real_data)
        synth_ac = compute_autocorrelation(self.synthetic_data)
        
        metrics['autocorr_real'] = real_ac
        metrics['autocorr_synth'] = synth_ac
        metrics['autocorr_diff'] = np.abs(real_ac - synth_ac)
        metrics['autocorr_similarity'] = 1 - metrics['autocorr_diff'] / (real_ac + 1e-8) if real_ac > 0 else 0
        
        # Cross-correlation between features
        def compute_cross_correlation(data):
            ccs = []
            n_sequences = min(50, len(data))
            
            for seq_idx in range(n_sequences):
                for i in range(min(3, data.shape[2])):
                    for j in range(i + 1, min(3, data.shape[2])):
                        series_i = data[seq_idx, :, i]
                        series_j = data[seq_idx, :, j]
                        try:
                            corr = np.corrcoef(series_i, series_j)[0, 1]
                            if not np.isnan(corr):
                                ccs.append(np.abs(corr))
                        except:
                            continue
            
            return np.mean(ccs) if ccs else 0
        
        real_cc = compute_cross_correlation(self.real_data)
        synth_cc = compute_cross_correlation(self.synthetic_data)
        
        metrics['crosscorr_real'] = real_cc
        metrics['crosscorr_synth'] = synth_cc
        metrics['crosscorr_similarity'] = 1 - np.abs(real_cc - synth_cc) / (real_cc + 1e-8) if real_cc > 0 else 0
        
        # Print results
        safe_print("Auto-correlation (Real):    {:.4f}".format(real_ac))
        safe_print("Auto-correlation (Synth):   {:.4f}".format(synth_ac))
        safe_print("Auto-correlation Similarity: {:.4f}".format(metrics['autocorr_similarity']))
        safe_print("Cross-correlation (Real):    {:.4f}".format(real_cc))
        safe_print("Cross-correlation (Synth):   {:.4f}".format(synth_cc))
        safe_print("Cross-correlation Similarity: {:.4f}".format(metrics['crosscorr_similarity']))
        
        return metrics
    
    def compute_feature_space_metrics(self):
        """Compute feature space similarity metrics"""
        safe_print("\n" + "="*50)
        safe_print("FEATURE SPACE METRICS")
        safe_print("="*50)
        
        metrics = {}
        
        # Flatten and sample data
        real_flat = self.real_data.reshape(-1, self.real_data.shape[2])
        synth_flat = self.synthetic_data.reshape(-1, self.synthetic_data.shape[2])
        
        n_samples = min(2000, len(real_flat), len(synth_flat))
        real_sample = real_flat[np.random.choice(len(real_flat), n_samples, replace=False)]
        synth_sample = synth_flat[np.random.choice(len(synth_flat), n_samples, replace=False)]
        
        combined = np.vstack([real_sample, synth_sample])
        labels = np.array([0] * n_samples + [1] * n_samples)
        
        # PCA analysis
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        real_pca = pca_result[:n_samples]
        synth_pca = pca_result[n_samples:]
        
        # Compute centroids
        real_centroid = np.mean(real_pca, axis=0)
        synth_centroid = np.mean(synth_pca, axis=0)
        
        metrics['centroid_distance'] = np.linalg.norm(real_centroid - synth_centroid)
        
        # Silhouette score
        try:
            silhouette = silhouette_score(pca_result, labels)
            metrics['silhouette_score'] = silhouette
        except:
            metrics['silhouette_score'] = 0.5
        
        # Print results
        safe_print("Centroid Distance:    {:.4f}".format(metrics['centroid_distance']))
        safe_print("Silhouette Score:     {:.4f}".format(metrics['silhouette_score']))
        
        return metrics, pca_result, labels
    
    def compute_overall_score(self, basic_metrics, dist_metrics, temp_metrics, feat_metrics):
        """Compute overall quality score"""
        safe_print("\n" + "="*50)
        safe_print("OVERALL QUALITY ASSESSMENT")
        safe_print("="*50)
        
        # Define weights for different metrics
        weights = {
            'statistical': 0.35,
            'distribution': 0.25,
            'temporal': 0.25,
            'feature_space': 0.15
        }
        
        # Compute component scores
        statistical_score = (
            max(basic_metrics['mean_correlation'], 0) * 0.5 +
            max(basic_metrics['std_correlation'], 0) * 0.3 +
            max(basic_metrics['range_similarity'], 0) * 0.2
        )
        
        distribution_score = (
            max(dist_metrics['ks_mean_pvalue'], 0) * 0.7 +
            (1 - min(dist_metrics['wasserstein_mean'], 1)) * 0.3
        )
        
        temporal_score = (
            max(temp_metrics['autocorr_similarity'], 0) * 0.6 +
            max(temp_metrics['crosscorr_similarity'], 0) * 0.4
        )
        
        feature_space_score = (
            (1 - min(feat_metrics['centroid_distance'] / 10, 1)) * 0.6 +
            max(feat_metrics['silhouette_score'], 0) * 0.4
        )
        
        # Compute overall score
        overall_score = (
            statistical_score * weights['statistical'] +
            distribution_score * weights['distribution'] +
            temporal_score * weights['temporal'] +
            feature_space_score * weights['feature_space']
        )
        
        # Create results dictionary
        results = {
            'overall_score': overall_score,
            'component_scores': {
                'statistical': statistical_score,
                'distribution': distribution_score,
                'temporal': temporal_score,
                'feature_space': feature_space_score
            },
            'detailed_metrics': {
                'basic': basic_metrics,
                'distribution': dist_metrics,
                'temporal': temp_metrics,
                'feature_space': feat_metrics
            }
        }
        
        # Print results
        safe_print("\nCOMPONENT SCORES:")
        safe_print("  Statistical:    {:.4f}".format(statistical_score))
        safe_print("  Distribution:   {:.4f}".format(distribution_score))
        safe_print("  Temporal:       {:.4f}".format(temporal_score))
        safe_print("  Feature Space:  {:.4f}".format(feature_space_score))
        
        safe_print("\nOVERALL QUALITY SCORE: {:.4f}".format(overall_score))
        
        if overall_score >= 0.8:
            safe_print("[SUCCESS] EXCELLENT - Synthetic data is highly realistic!")
        elif overall_score >= 0.6:
            safe_print("[SUCCESS] GOOD - Synthetic data is realistic")
        elif overall_score >= 0.4:
            safe_print("[WARNING] FAIR - Synthetic data has some issues")
        else:
            safe_print("[FAILED] POOR - Synthetic data needs improvement")
        
        return results
    
    def create_visualizations(self, pca_result, labels, output_dir='outputs/evaluation_plots'):
        """Create comprehensive visualizations"""
        safe_print("\n" + "="*50)
        safe_print("CREATING VISUALIZATIONS")
        safe_print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        n_samples = len(labels) // 2
        real_pca = pca_result[:n_samples]
        synth_pca = pca_result[n_samples:]
        
        # 1. PCA Scatter Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', s=10)
        plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label='Synthetic', s=10)
        plt.title('PCA of Real vs Synthetic Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'pca_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Time Series Comparison
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        n_plots = min(6, len(self.real_data), len(self.synthetic_data))
        plot_indices = np.random.choice(len(self.real_data), n_plots, replace=False)
        
        for i, idx in enumerate(plot_indices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            # Plot first 3 features
            for feat in range(min(3, self.real_data.shape[2])):
                real_series = self.real_data[idx, :, feat]
                synth_series = self.synthetic_data[idx, :, feat]
                
                ax.plot(real_series, alpha=0.7, linewidth=2, 
                       label=f'Real F{feat}' if i == 0 else "")
                ax.plot(synth_series, alpha=0.7, linestyle='--', linewidth=2,
                       label=f'Synth F{feat}' if i == 0 else "")
            
            ax.set_title(f'Sample {idx}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Distribution Comparison
        n_features = min(6, self.real_data.shape[2])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for feat in range(n_features):
            if feat >= len(axes):
                break
                
            ax = axes[feat]
            
            real_flat = self.real_data[:, :, feat].flatten()
            synth_flat = self.synthetic_data[:, :, feat].flatten()
            
            # Sample for plotting
            if len(real_flat) > 10000:
                real_flat = real_flat[np.random.choice(len(real_flat), 10000, replace=False)]
                synth_flat = synth_flat[np.random.choice(len(synth_flat), 10000, replace=False)]
            
            ax.hist(real_flat, bins=50, alpha=0.5, density=True, label='Real')
            ax.hist(synth_flat, bins=50, alpha=0.5, density=True, label='Synthetic')
            ax.set_title(f'Feature {feat} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Statistical Summary Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean comparison
        real_means = np.mean(self.real_data, axis=(0, 1))
        synth_means = np.mean(self.synthetic_data, axis=(0, 1))
        
        axes[0, 0].scatter(real_means, synth_means, alpha=0.6)
        axes[0, 0].plot([real_means.min(), real_means.max()], 
                       [real_means.min(), real_means.max()], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Real Means')
        axes[0, 0].set_ylabel('Synthetic Means')
        axes[0, 0].set_title('Feature Means Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Std comparison
        real_stds = np.std(self.real_data, axis=(0, 1))
        synth_stds = np.std(self.synthetic_data, axis=(0, 1))
        
        axes[0, 1].scatter(real_stds, synth_stds, alpha=0.6)
        axes[0, 1].plot([real_stds.min(), real_stds.max()], 
                       [real_stds.min(), real_stds.max()], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Real Stds')
        axes[0, 1].set_ylabel('Synthetic Stds')
        axes[0, 1].set_title('Feature Standard Deviations Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation matrix (real)
        real_corr = np.corrcoef(self.real_data.reshape(-1, self.real_data.shape[2]).T)
        im1 = axes[1, 0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Real Data Correlation Matrix')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # Correlation matrix (synthetic)
        synth_corr = np.corrcoef(self.synthetic_data.reshape(-1, self.synthetic_data.shape[2]).T)
        im2 = axes[1, 1].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Synthetic Data Correlation Matrix')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statistical_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        safe_print("[PLOT] Visualizations saved to: {}".format(output_dir))
    
    def save_results(self, results, output_dir='outputs/evaluation_results'):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results as JSON
        results_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        safe_print("[SAVE] Results saved to: {}".format(results_path))
        
        # Also save a summary text file
        summary_path = os.path.join(output_dir, f'evaluation_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write("AIR QUALITY SYNTHETIC DATA EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Quality Score: {results['overall_score']:.4f}\n\n")
            
            f.write("COMPONENT SCORES:\n")
            for component, score in results['component_scores'].items():
                f.write(f"  {component}: {score:.4f}\n")
            
            f.write("\nDETAILED METRICS:\n")
            f.write("  Basic Statistics:\n")
            for key, value in results['detailed_metrics']['basic'].items():
                if isinstance(value, float):
                    f.write(f"    {key}: {value:.4f}\n")
            
            f.write("\n  Distribution:\n")
            for key, value in results['detailed_metrics']['distribution'].items():
                if isinstance(value, float):
                    f.write(f"    {key}: {value:.4f}\n")
            
            f.write("\n  Temporal Patterns:\n")
            for key, value in results['detailed_metrics']['temporal'].items():
                if isinstance(value, float):
                    f.write(f"    {key}: {value:.4f}\n")
            
            f.write("\n  Feature Space:\n")
            for key, value in results['detailed_metrics']['feature_space'].items():
                if isinstance(value, float):
                    f.write(f"    {key}: {value:.4f}\n")
        
        safe_print("[SAVE] Summary saved to: {}".format(summary_path))
        
        return results_path, summary_path

def main():
    safe_print("\n" + "="*60)
    safe_print("AIR QUALITY SYNTHETIC DATA EVALUATION")
    safe_print("="*60)
    
    # Initialize evaluator
    evaluator = SimpleAirQualityEvaluator()
    
    # Load data
    if not evaluator.load_data():
        safe_print("[FAILED] Failed to load data. Exiting...")
        sys.exit(1)
    
    # Compute metrics
    safe_print("\nComputing evaluation metrics...")
    
    # 1. Basic statistical metrics
    basic_metrics = evaluator.compute_basic_metrics()
    
    # 2. Distribution metrics
    dist_metrics = evaluator.compute_distribution_metrics()
    
    # 3. Temporal metrics
    temp_metrics = evaluator.compute_temporal_metrics()
    
    # 4. Feature space metrics
    feat_metrics, pca_result, labels = evaluator.compute_feature_space_metrics()
    
    # 5. Overall score
    results = evaluator.compute_overall_score(basic_metrics, dist_metrics, temp_metrics, feat_metrics)
    
    # 6. Create visualizations
    evaluator.create_visualizations(pca_result, labels)
    
    # 7. Save results
    evaluator.save_results(results)
    
    safe_print("\n" + "="*60)
    safe_print("EVALUATION COMPLETE!")
    safe_print("="*60)

if __name__ == "__main__":
    main()