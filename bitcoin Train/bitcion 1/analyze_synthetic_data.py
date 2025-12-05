import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OptimizedSyntheticDataAnalyzer:
    def __init__(self, real_data_path, synthetic_scaled_path, synthetic_inverse_path, feature_names):
        self.real_data = np.load(real_data_path)
        self.synthetic_scaled = np.load(synthetic_scaled_path)
        self.synthetic_inverse = np.load(synthetic_inverse_path)
        self.feature_names = feature_names
        
        print("📊 DATA LOADED SUCCESSFULLY")
        print(f"Real data shape: {self.real_data.shape}")
        print(f"Synthetic scaled shape: {self.synthetic_scaled.shape}")
        print(f"Synthetic inverse shape: {self.synthetic_inverse.shape}")
        print(f"Features: {len(feature_names)}")
        
    def memory_efficient_correlation(self, data, sample_size=10000):
        """Calculate correlation matrix with memory efficiency"""
        n_samples, seq_len, n_features = data.shape
        
        # Sample data to avoid memory issues
        if n_samples * seq_len > sample_size:
            # Flatten and sample
            data_flat = data.reshape(-1, n_features)
            sample_idx = np.random.choice(data_flat.shape[0], sample_size, replace=False)
            data_sampled = data_flat[sample_idx]
        else:
            data_sampled = data.reshape(-1, n_features)
        
        return np.corrcoef(data_sampled, rowvar=False)
    
    def basic_statistical_analysis(self):
        """Memory-efficient statistical analysis"""
        print("\n" + "="*80)
        print("BASIC STATISTICAL ANALYSIS")
        print("="*80)
        
        # Sample data for analysis to save memory
        real_sample = self.real_data[:1000]  # Use first 1000 sequences
        synth_sample = self.synthetic_scaled[:1000]
        
        real_flat = real_sample.reshape(-1, real_sample.shape[2])
        synth_flat = synth_sample.reshape(-1, synth_sample.shape[2])
        
        print("\n📈 FEATURE-WISE STATISTICAL COMPARISON (Sampled)")
        print("-" * 120)
        print(f"{'Feature':<15} {'Dataset':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 120)
        
        for i, feature in enumerate(self.feature_names):
            real_feat = real_flat[:, i]
            synth_feat = synth_flat[:, i]
            
            print(f"{feature:<15} {'REAL':<10} {np.mean(real_feat):>10.4f} {np.std(real_feat):>10.4f} "
                  f"{np.min(real_feat):>10.4f} {np.max(real_feat):>10.4f}")
            print(f"{'':<15} {'SYNTH':<10} {np.mean(synth_feat):>10.4f} {np.std(synth_feat):>10.4f} "
                  f"{np.min(synth_feat):>10.4f} {np.max(synth_feat):>10.4f}")
            print("-" * 120)
    
    def distribution_comparison(self, save_path=None):
        """Compare distributions with sampling"""
        print("\n" + "="*80)
        print("DISTRIBUTION COMPARISON")
        print("="*80)
        
        # Sample data
        real_sample = self.real_data[:500]  # Use 500 sequences
        synth_sample = self.synthetic_scaled[:500]
        
        real_flat = real_sample.reshape(-1, real_sample.shape[2])
        synth_flat = synth_sample.reshape(-1, synth_sample.shape[2])
        
        # Select key features
        key_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Volatility']
        features_to_plot = [f for f in key_features if f in self.feature_names]
        
        n_features = len(features_to_plot)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, feature_name in enumerate(features_to_plot):
            if idx >= len(axes):
                break
                
            feature_idx = self.feature_names.index(feature_name)
            
            real_feature = real_flat[:, feature_idx]
            synth_feature = synth_flat[:, feature_idx]
            
            # Remove outliers
            def remove_outliers(data):
                q1, q99 = np.percentile(data, [1, 99])
                return data[(data >= q1) & (data <= q99)]
            
            real_clean = remove_outliers(real_feature)
            synth_clean = remove_outliers(synth_feature)
            
            # Plot histograms
            axes[idx].hist(real_clean, bins=50, alpha=0.7, label='Real', density=True, 
                          color='blue', edgecolor='black', linewidth=0.5)
            axes[idx].hist(synth_clean, bins=50, alpha=0.7, label='Synthetic', density=True, 
                          color='red', edgecolor='black', linewidth=0.5)
            
            axes[idx].set_title(f'{feature_name} Distribution', fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlabel('Value', fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
        
        # Hide empty subplots
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Distribution plots saved: {save_path}")
        
        plt.show()
    
    def correlation_analysis(self, save_path=None):
        """Memory-efficient correlation analysis"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Use sampled data for correlation
        real_corr = self.memory_efficient_correlation(self.real_data, sample_size=5000)
        synth_corr = self.memory_efficient_correlation(self.synthetic_scaled, sample_size=5000)
        
        # Correlation difference
        corr_diff = np.abs(real_corr - synth_corr)
        
        print(f"Real correlation matrix shape: {real_corr.shape}")
        print(f"Synthetic correlation matrix shape: {synth_corr.shape}")
        print(f"Average absolute correlation difference: {np.mean(corr_diff):.4f}")
        print(f"Maximum correlation difference: {np.max(corr_diff):.4f}")
        
        # Plot top correlations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Real correlation
        im1 = axes[0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title('Real Data Correlation', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(len(self.feature_names)))
        axes[0].set_yticks(range(len(self.feature_names)))
        axes[0].set_xticklabels(self.feature_names, rotation=45, ha='right')
        axes[0].set_yticklabels(self.feature_names)
        plt.colorbar(im1, ax=axes[0])
        
        # Synthetic correlation
        im2 = axes[1].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_title('Synthetic Data Correlation', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(self.feature_names)))
        axes[1].set_yticks(range(len(self.feature_names)))
        axes[1].set_xticklabels(self.feature_names, rotation=45, ha='right')
        axes[1].set_yticklabels(self.feature_names)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Correlation plots saved: {save_path}")
        
        plt.show()
        
        return real_corr, synth_corr, corr_diff
    
    def financial_metrics_analysis(self):
        """Analyze key financial metrics with sampling"""
        print("\n" + "="*80)
        print("FINANCIAL METRICS ANALYSIS")
        print("="*80)
        
        # Sample synthetic data for analysis
        sample_size = min(1000, self.synthetic_inverse.shape[0])
        synth_sample = self.synthetic_inverse[:sample_size]
        
        # Get indices for financial features
        feature_indices = {}
        for name in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if name in self.feature_names:
                feature_indices[name] = self.feature_names.index(name)
        
        if not all(k in feature_indices for k in ['Open', 'High', 'Low', 'Close']):
            print("Missing required price features for financial analysis")
            return
        
        # Calculate financial metrics
        open_idx = feature_indices['Open']
        high_idx = feature_indices['High']
        low_idx = feature_indices['Low']
        close_idx = feature_indices['Close']
        volume_idx = feature_indices.get('Volume')
        
        returns = []
        high_low_spreads = []
        volumes = []
        
        for i in range(sample_size):
            close_prices = synth_sample[i, :, close_idx]
            high_prices = synth_sample[i, :, high_idx]
            low_prices = synth_sample[i, :, low_idx]
            
            # Calculate returns
            if len(close_prices) > 1:
                sequence_returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
                returns.extend(sequence_returns)
            
            # Calculate high-low spreads
            spreads = (high_prices - low_prices) / close_prices
            high_low_spreads.extend(spreads)
            
            # Volume data
            if volume_idx is not None:
                volume_data = synth_sample[i, :, volume_idx]
                volumes.extend(volume_data)
        
        returns = np.array(returns)
        spreads = np.array(high_low_spreads)
        volumes = np.array(volumes)
        
        # Remove extreme outliers
        def remove_extreme_outliers(data, threshold=0.01):
            if len(data) == 0:
                return data
            q1, q99 = np.percentile(data, [threshold, 100-threshold])
            return data[(data >= q1) & (data <= q99)]
        
        returns_clean = remove_extreme_outliers(returns)
        spreads_clean = remove_extreme_outliers(spreads)
        volumes_clean = remove_extreme_outliers(volumes)
        
        print("💰 SYNTHETIC BITCOIN FINANCIAL METRICS")
        print("-" * 50)
        if len(returns_clean) > 0:
            print(f"Average Daily Return: {np.mean(returns_clean)*100:.2f}%")
            print(f"Return Volatility: {np.std(returns_clean)*100:.2f}%")
            print(f"Positive Returns: {(returns_clean > 0).sum() / len(returns_clean)*100:.1f}%")
            print(f"Maximum Return: {np.max(returns_clean)*100:.2f}%")
            print(f"Minimum Return: {np.min(returns_clean)*100:.2f}%")
        
        if len(spreads_clean) > 0:
            print(f"Average High-Low Spread: {np.mean(spreads_clean)*100:.2f}%")
        
        if len(volumes_clean) > 0:
            print(f"Average Volume: ${np.mean(volumes_clean):.2f}")
            print(f"Volume Volatility: ${np.std(volumes_clean):.2f}")
        
        # Plot distributions
        if len(returns_clean) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].hist(returns_clean * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0].set_title('Synthetic Bitcoin Returns', fontweight='bold')
            axes[0].set_xlabel('Daily Return (%)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            if len(spreads_clean) > 0:
                axes[1].hist(spreads_clean * 100, bins=50, alpha=0.7, color='orange', edgecolor='black')
                axes[1].set_title('High-Low Spreads', fontweight='bold')
                axes[1].set_xlabel('Spread (%)')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return {
            'returns': returns_clean,
            'spreads': spreads_clean,
            'volumes': volumes_clean
        }
    
    def temporal_pattern_analysis(self, save_path=None):
        """Analyze temporal patterns with sampling"""
        print("\n" + "="*80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*80)
        
        # Sample sequences
        n_sequences = 3
        real_sample_idx = np.random.randint(0, min(100, self.real_data.shape[0]), n_sequences)
        synth_sample_idx = np.random.randint(0, min(100, self.synthetic_scaled.shape[0]), n_sequences)
        
        # Focus on price features
        price_features = ['Open', 'High', 'Low', 'Close']
        price_indices = [self.feature_names.index(f) for f in price_features if f in self.feature_names]
        
        if not price_indices:
            print("No price features found for temporal analysis")
            return
        
        n_price_features = len(price_indices)
        fig, axes = plt.subplots(n_price_features, 2, figsize=(15, 3*n_price_features))
        
        if n_price_features == 1:
            axes = axes.reshape(1, -1)
        
        for i, price_idx in enumerate(price_indices):
            feature_name = self.feature_names[price_idx]
            
            # Real data
            for j, seq_idx in enumerate(real_sample_idx):
                real_sequence = self.real_data[seq_idx, :, price_idx]
                axes[i, 0].plot(real_sequence, alpha=0.7, linewidth=2, label=f'Seq {j+1}')
            axes[i, 0].set_title(f'Real {feature_name}', fontweight='bold')
            axes[i, 0].set_ylabel('Scaled Value')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            
            # Synthetic data
            for j, seq_idx in enumerate(synth_sample_idx):
                synth_sequence = self.synthetic_scaled[seq_idx, :, price_idx]
                axes[i, 1].plot(synth_sequence, alpha=0.7, linewidth=2, label=f'Seq {j+1}')
            axes[i, 1].set_title(f'Synthetic {feature_name}', fontweight='bold')
            axes[i, 1].set_ylabel('Scaled Value')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Temporal patterns saved: {save_path}")
        
        plt.show()
    
    def quality_assessment(self):
        """Memory-efficient quality assessment"""
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT")
        print("="*80)
        
        # Use sampled data
        real_sample = self.real_data[:1000]
        synth_sample = self.synthetic_scaled[:1000]
        
        real_flat = real_sample.reshape(-1, real_sample.shape[2])
        synth_flat = synth_sample.reshape(-1, synth_sample.shape[2])
        
        quality_scores = []
        
        print("🔍 FEATURE-LEVEL QUALITY SCORES")
        print("-" * 60)
        print(f"{'Feature':<15} {'Mean Diff':<12} {'Std Diff':<12} {'Score':<10}")
        print("-" * 60)
        
        for i, feature in enumerate(self.feature_names):
            real_feat = real_flat[:, i]
            synth_feat = synth_flat[:, i]
            
            # Remove outliers
            def remove_outliers(data):
                q1, q99 = np.percentile(data, [1, 99])
                return data[(data >= q1) & (data <= q99)]
            
            real_clean = remove_outliers(real_feat)
            synth_clean = remove_outliers(synth_feat)
            
            # Calculate differences
            mean_diff = abs(np.mean(synth_clean) - np.mean(real_clean))
            std_diff = abs(np.std(synth_clean) - np.std(real_clean))
            
            # Calculate score (0-1, higher is better)
            mean_score = max(0, 1 - mean_diff)
            std_score = max(0, 1 - std_diff)
            feature_score = (mean_score + std_score) / 2
            
            quality_scores.append(feature_score)
            
            print(f"{feature:<15} {mean_diff:>10.4f} {std_diff:>10.4f} {feature_score:>9.1%}")
        
        overall_quality = np.mean(quality_scores)
        
        print("-" * 60)
        print(f"{'OVERALL QUALITY':<15} {'':<22} {overall_quality:>9.1%}")
        
        # Quality interpretation
        if overall_quality >= 0.9:
            rating = "🎉 EXCELLENT"
            color = "green"
        elif overall_quality >= 0.8:
            rating = "👍 VERY GOOD"
            color = "blue"
        elif overall_quality >= 0.7:
            rating = "✅ GOOD"
            color = "orange"
        elif overall_quality >= 0.6:
            rating = "⚠️  ACCEPTABLE"
            color = "red"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
            color = "darkred"
        
        print(f"\nRATING: {rating}")
        
        # Plot quality scores
        plt.figure(figsize=(12, 6))
        features_short = [f[:12] for f in self.feature_names]  # Shorten names for display
        
        bars = plt.bar(features_short, quality_scores, color='skyblue', edgecolor='black')
        
        # Color bars based on score
        for j, bar in enumerate(bars):
            if quality_scores[j] >= 0.9:
                bar.set_color('green')
            elif quality_scores[j] >= 0.8:
                bar.set_color('lightgreen')
            elif quality_scores[j] >= 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.axhline(y=overall_quality, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_quality:.1%}')
        plt.title('Feature-wise Quality Scores', fontsize=16, fontweight='bold')
        plt.ylabel('Quality Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        
        return overall_quality
    
    def generate_report(self, output_dir="outputs/analysis_optimized"):
        """Generate comprehensive optimized report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 GENERATING OPTIMIZED ANALYSIS REPORT...")
        
        # Run analyses
        self.basic_statistical_analysis()
        self.distribution_comparison(save_path=os.path.join(output_dir, "distributions.png"))
        real_corr, synth_corr, corr_diff = self.correlation_analysis(save_path=os.path.join(output_dir, "correlations.png"))
        self.temporal_pattern_analysis(save_path=os.path.join(output_dir, "temporal_patterns.png"))
        financial_metrics = self.financial_metrics_analysis()
        overall_quality = self.quality_assessment()
        
        print(f"\n📋 REPORT SUMMARY")
        print("=" * 50)
        print(f"Overall Quality Score: {overall_quality:.1%}")
        print(f"Real Data Sequences: {self.real_data.shape[0]:,}")
        print(f"Synthetic Data Sequences: {self.synthetic_scaled.shape[0]:,}")
        print(f"Data Expansion: {self.synthetic_scaled.shape[0] / self.real_data.shape[0]:.1f}x")
        
        if financial_metrics and 'returns' in financial_metrics:
            returns = financial_metrics['returns']
            if len(returns) > 0:
                print(f"Average Return: {np.mean(returns)*100:.2f}%")
                print(f"Return Volatility: {np.std(returns)*100:.2f}%")
        
        print(f"\n💾 REPORT SAVED TO: {output_dir}/")
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")

# Main execution
if __name__ == "__main__":
    # Configuration
    REAL_DATA_PATH = "data/processed/crypto/train.npy"
    SYNTHETIC_SCALED_PATH = "outputs/synth_fixed/synthetic_scaled_20251128_225900.npy"
    SYNTHETIC_INVERSE_PATH = "outputs/synth_fixed/synthetic_inverse_20251128_225900.npy"
    
    # Load feature names
    feature_names_path = "data/processed/crypto/features.txt"
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            FEATURE_NAMES = [line.strip() for line in f.readlines()]
    else:
        FEATURE_NAMES = [f"feature_{i}" for i in range(14)]
    
    print("🔍 OPTIMIZED SYNTHETIC BITCOIN DATA ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = OptimizedSyntheticDataAnalyzer(
        REAL_DATA_PATH, 
        SYNTHETIC_SCALED_PATH, 
        SYNTHETIC_INVERSE_PATH, 
        FEATURE_NAMES
    )
    
    # Generate report
    analyzer.generate_report()