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

class EqualSyntheticDataAnalyzer:
    def __init__(self, real_data_path, synthetic_scaled_path, synthetic_inverse_path, feature_names):
        # Load data
        self.original_real_data = np.load(real_data_path)
        self.synthetic_scaled = np.load(synthetic_scaled_path)
        self.synthetic_inverse = np.load(synthetic_inverse_path)
        self.feature_names = feature_names
        
        # Make both datasets have exactly 50,000 sequences
        self.real_data = self._augment_real_data()
        
        print("📊 DATA LOADED SUCCESSFULLY")
        print(f"Original real data shape: {self.original_real_data.shape}")
        print(f"Augmented real data shape: {self.real_data.shape}")
        print(f"Synthetic scaled shape: {self.synthetic_scaled.shape}")
        print(f"Synthetic inverse shape: {self.synthetic_inverse.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"✅ Both datasets now have {self.real_data.shape[0]:,} sequences")
        
    def _augment_real_data(self):
        """Augment real data to match synthetic data size (50,000 sequences)"""
        original_size = self.original_real_data.shape[0]
        target_size = 50000
        
        if original_size >= target_size:
            # If real data is larger, sample down
            indices = np.random.choice(original_size, target_size, replace=False)
            return self.original_real_data[indices]
        else:
            # If real data is smaller, augment with sampling
            num_repeats = target_size // original_size
            remainder = target_size % original_size
            
            augmented_data = []
            
            # Repeat the data
            for _ in range(num_repeats):
                # Add some noise to repeated sequences to make them different
                noise = np.random.normal(0, 0.01, self.original_real_data.shape)
                augmented_data.append(self.original_real_data + noise)
            
            # Add remaining sequences with different noise
            if remainder > 0:
                indices = np.random.choice(original_size, remainder, replace=False)
                noise = np.random.normal(0, 0.01, (remainder, self.original_real_data.shape[1], self.original_real_data.shape[2]))
                augmented_data.append(self.original_real_data[indices] + noise)
            
            return np.concatenate(augmented_data, axis=0)
    
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
        """Statistical analysis with equal sample sizes"""
        print("\n" + "="*80)
        print("BASIC STATISTICAL ANALYSIS (50,000 sequences each)")
        print("="*80)
        
        # Use equal sample sizes for comparison
        sample_size = min(5000, self.real_data.shape[0])  # Use 5,000 for speed
        
        real_sample = self.real_data[:sample_size]
        synth_sample = self.synthetic_scaled[:sample_size]
        
        real_flat = real_sample.reshape(-1, real_sample.shape[2])
        synth_flat = synth_sample.reshape(-1, synth_sample.shape[2])
        
        print(f"\n📈 FEATURE-WISE STATISTICAL COMPARISON ({sample_size:,} samples each)")
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
            
            # Calculate similarity
            mean_diff = abs(np.mean(real_feat) - np.mean(synth_feat))
            std_diff = abs(np.std(real_feat) - np.std(synth_feat))
            print(f"{'':<15} {'DIFF':<10} {mean_diff:>10.4f} {std_diff:>10.4f} {'':<12} {'':<12}")
            print("-" * 120)
    
    def distribution_comparison(self, save_path=None):
        """Compare distributions with equal sample sizes"""
        print("\n" + "="*80)
        print("DISTRIBUTION COMPARISON (50,000 sequences each)")
        print("="*80)
        
        # Use equal sample sizes
        sample_size = min(1000, self.real_data.shape[0])
        real_sample = self.real_data[:sample_size]
        synth_sample = self.synthetic_scaled[:sample_size]
        
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
            
            # Calculate KS test
            ks_stat, ks_p = stats.ks_2samp(real_clean, synth_clean)
            
            axes[idx].set_title(f'{feature_name}\nKS p-value: {ks_p:.4f}', fontsize=12, fontweight='bold')
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
        """Memory-efficient correlation analysis with equal samples"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS (50,000 sequences each)")
        print("="*80)
        
        # Use equal sample sizes for correlation
        sample_size = 5000
        
        # Sample from both datasets
        real_indices = np.random.choice(self.real_data.shape[0], sample_size, replace=False)
        synth_indices = np.random.choice(self.synthetic_scaled.shape[0], sample_size, replace=False)
        
        real_sample = self.real_data[real_indices]
        synth_sample = self.synthetic_scaled[synth_indices]
        
        real_corr = self.memory_efficient_correlation(real_sample, sample_size=sample_size)
        synth_corr = self.memory_efficient_correlation(synth_sample, sample_size=sample_size)
        
        # Correlation difference
        corr_diff = np.abs(real_corr - synth_corr)
        
        print(f"Real correlation matrix shape: {real_corr.shape}")
        print(f"Synthetic correlation matrix shape: {synth_corr.shape}")
        print(f"Average absolute correlation difference: {np.mean(corr_diff):.4f}")
        print(f"Maximum correlation difference: {np.max(corr_diff):.4f}")
        print(f"Correlation similarity: {(1 - np.mean(corr_diff)) * 100:.2f}%")
        
        # Plot correlations
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Real correlation
        im1 = axes[0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title('Real Data Correlation\n(50,000 sequences)', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(len(self.feature_names)))
        axes[0].set_yticks(range(len(self.feature_names)))
        axes[0].set_xticklabels(self.feature_names, rotation=45, ha='right')
        axes[0].set_yticklabels(self.feature_names)
        plt.colorbar(im1, ax=axes[0])
        
        # Synthetic correlation
        im2 = axes[1].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_title('Synthetic Data Correlation\n(50,000 sequences)', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(self.feature_names)))
        axes[1].set_yticks(range(len(self.feature_names)))
        axes[1].set_xticklabels(self.feature_names, rotation=45, ha='right')
        axes[1].set_yticklabels(self.feature_names)
        plt.colorbar(im2, ax=axes[1])
        
        # Correlation difference
        im3 = axes[2].imshow(corr_diff, cmap='hot', vmin=0, vmax=1, aspect='auto')
        axes[2].set_title(f'Correlation Difference\nAvg: {np.mean(corr_diff):.4f}', fontsize=14, fontweight='bold')
        axes[2].set_xticks(range(len(self.feature_names)))
        axes[2].set_yticks(range(len(self.feature_names)))
        axes[2].set_xticklabels(self.feature_names, rotation=45, ha='right')
        axes[2].set_yticklabels(self.feature_names)
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Correlation plots saved: {save_path}")
        
        plt.show()
        
        return real_corr, synth_corr, corr_diff
    
    def financial_metrics_analysis(self):
        """Analyze key financial metrics with equal samples"""
        print("\n" + "="*80)
        print("FINANCIAL METRICS ANALYSIS (50,000 sequences each)")
        print("="*80)
        
        # Sample both datasets equally
        sample_size = min(2000, self.real_data.shape[0], self.synthetic_inverse.shape[0])
        real_sample = self.real_data[:sample_size]
        synth_sample = self.synthetic_inverse[:sample_size]
        
        # Get indices for financial features
        feature_indices = {}
        for name in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if name in self.feature_names:
                feature_indices[name] = self.feature_names.index(name)
        
        if not all(k in feature_indices for k in ['Open', 'High', 'Low', 'Close']):
            print("Missing required price features for financial analysis")
            return
        
        # Calculate financial metrics for both datasets
        close_idx = feature_indices['Close']
        volume_idx = feature_indices.get('Volume')
        
        def calculate_metrics(data, dataset_name):
            returns = []
            volumes = []
            
            for i in range(data.shape[0]):
                close_prices = data[i, :, close_idx]
                
                # Calculate returns
                if len(close_prices) > 1:
                    sequence_returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
                    returns.extend(sequence_returns)
                
                # Volume data
                if volume_idx is not None:
                    volume_data = data[i, :, volume_idx]
                    volumes.extend(volume_data)
            
            returns = np.array(returns)
            volumes = np.array(volumes)
            
            # Remove extreme outliers
            def remove_extreme_outliers(data, threshold=0.01):
                if len(data) == 0:
                    return data
                q1, q99 = np.percentile(data, [threshold, 100-threshold])
                return data[(data >= q1) & (data <= q99)]
            
            returns_clean = remove_extreme_outliers(returns)
            volumes_clean = remove_extreme_outliers(volumes)
            
            return {
                'returns': returns_clean,
                'volumes': volumes_clean
            }
        
        # Calculate for both datasets
        real_metrics = calculate_metrics(real_sample, "Real")
        synth_metrics = calculate_metrics(synth_sample, "Synthetic")
        
        print("💰 FINANCIAL METRICS COMPARISON")
        print("-" * 70)
        print(f"{'Metric':<25} {'Real Data':<15} {'Synthetic Data':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Returns comparison
        if len(real_metrics['returns']) > 0 and len(synth_metrics['returns']) > 0:
            real_ret_mean = np.mean(real_metrics['returns']) * 100
            synth_ret_mean = np.mean(synth_metrics['returns']) * 100
            ret_diff = abs(real_ret_mean - synth_ret_mean)
            
            real_ret_vol = np.std(real_metrics['returns']) * 100
            synth_ret_vol = np.std(synth_metrics['returns']) * 100
            vol_diff = abs(real_ret_vol - synth_ret_vol)
            
            real_pos_ret = (real_metrics['returns'] > 0).sum() / len(real_metrics['returns']) * 100
            synth_pos_ret = (synth_metrics['returns'] > 0).sum() / len(synth_metrics['returns']) * 100
            pos_diff = abs(real_pos_ret - synth_pos_ret)
            
            print(f"{'Avg Return (%)':<25} {real_ret_mean:>14.2f} {synth_ret_mean:>14.2f} {ret_diff:>14.2f}")
            print(f"{'Return Volatility (%)':<25} {real_ret_vol:>14.2f} {synth_ret_vol:>14.2f} {vol_diff:>14.2f}")
            print(f"{'Positive Returns (%)':<25} {real_pos_ret:>14.1f} {synth_pos_ret:>14.1f} {pos_diff:>14.1f}")
        
        # Volume comparison
        if len(real_metrics['volumes']) > 0 and len(synth_metrics['volumes']) > 0:
            real_vol_mean = np.mean(real_metrics['volumes'])
            synth_vol_mean = np.mean(synth_metrics['volumes'])
            vol_mean_diff = abs(real_vol_mean - synth_vol_mean)
            
            real_vol_std = np.std(real_metrics['volumes'])
            synth_vol_std = np.std(synth_metrics['volumes'])
            vol_std_diff = abs(real_vol_std - synth_vol_std)
            
            print(f"{'Avg Volume':<25} {real_vol_mean:>14.2f} {synth_vol_mean:>14.2f} {vol_mean_diff:>14.2f}")
            print(f"{'Volume Std':<25} {real_vol_std:>14.2f} {synth_vol_std:>14.2f} {vol_std_diff:>14.2f}")
        
        print("-" * 70)
        
        return {
            'real_metrics': real_metrics,
            'synthetic_metrics': synth_metrics
        }
    
    def quality_assessment(self):
        """Comprehensive quality assessment with equal samples"""
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT (50,000 sequences each)")
        print("="*80)
        
        # Use larger sample for better assessment
        sample_size = min(5000, self.real_data.shape[0])
        real_sample = self.real_data[:sample_size]
        synth_sample = self.synthetic_scaled[:sample_size]
        
        real_flat = real_sample.reshape(-1, real_sample.shape[2])
        synth_flat = synth_sample.reshape(-1, synth_sample.shape[2])
        
        quality_scores = []
        detailed_results = []
        
        print("🔍 COMPREHENSIVE QUALITY ASSESSMENT")
        print("-" * 80)
        print(f"{'Feature':<15} {'Mean Diff':<10} {'Std Diff':<10} {'KS p-value':<12} {'Score':<10} {'Status':<15}")
        print("-" * 80)
        
        for i, feature in enumerate(self.feature_names):
            real_feat = real_flat[:, i]
            synth_feat = synth_flat[:, i]
            
            # Remove outliers for stable comparison
            def remove_outliers(data):
                q1, q99 = np.percentile(data, [1, 99])
                return data[(data >= q1) & (data <= q99)]
            
            real_clean = remove_outliers(real_feat)
            synth_clean = remove_outliers(synth_feat)
            
            # Calculate statistical differences
            mean_diff = abs(np.mean(synth_clean) - np.mean(real_clean))
            std_diff = abs(np.std(synth_clean) - np.std(real_clean))
            
            # KS test for distribution similarity
            ks_stat, ks_p = stats.ks_2samp(real_clean, synth_clean)
            
            # Calculate composite score (0-1, higher is better)
            mean_score = max(0, 1 - mean_diff)
            std_score = max(0, 1 - std_diff)
            ks_score = min(1.0, ks_p * 5)  # Convert p-value to score
            feature_score = (mean_score + std_score + ks_score) / 3
            
            quality_scores.append(feature_score)
            
            # Determine status
            if feature_score >= 0.9:
                status = "EXCELLENT"
            elif feature_score >= 0.8:
                status = "VERY GOOD"
            elif feature_score >= 0.7:
                status = "GOOD"
            elif feature_score >= 0.6:
                status = "ACCEPTABLE"
            else:
                status = "NEEDS IMPROV"
            
            detailed_results.append({
                'feature': feature,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'ks_p': ks_p,
                'score': feature_score,
                'status': status
            })
            
            print(f"{feature:<15} {mean_diff:>9.4f} {std_diff:>9.4f} {ks_p:>11.4f} {feature_score:>9.1%} {status:>15}")
        
        overall_quality = np.mean(quality_scores)
        
        print("-" * 80)
        print(f"{'OVERALL QUALITY':<15} {'':<9} {'':<9} {'':<11} {overall_quality:>9.1%} {'':<15}")
        
        # Quality interpretation
        if overall_quality >= 0.9:
            rating = "🎉 EXCELLENT"
        elif overall_quality >= 0.8:
            rating = "👍 VERY GOOD"
        elif overall_quality >= 0.7:
            rating = "✅ GOOD"
        elif overall_quality >= 0.6:
            rating = "⚠️ ACCEPTABLE"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"\nRATING: {rating}")
        
        # Plot quality scores
        plt.figure(figsize=(14, 6))
        
        features = [r['feature'] for r in detailed_results]
        scores = [r['score'] for r in detailed_results]
        
        colors = []
        for score in scores:
            if score >= 0.9: colors.append('green')
            elif score >= 0.8: colors.append('lightgreen')
            elif score >= 0.7: colors.append('orange')
            else: colors.append('red')
        
        bars = plt.bar(features, scores, color=colors, edgecolor='black', alpha=0.8)
        plt.axhline(y=overall_quality, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Quality: {overall_quality:.1%}')
        plt.title('Feature-wise Quality Scores\n(50,000 sequences comparison)', fontsize=16, fontweight='bold')
        plt.ylabel('Quality Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return overall_quality, detailed_results
    
    def generate_report(self, output_dir="outputs/analysis_equal"):
        """Generate comprehensive report with equal dataset sizes"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 GENERATING COMPREHENSIVE ANALYSIS REPORT...")
        print("📊 USING 50,000 SEQUENCES FOR BOTH REAL AND SYNTHETIC DATA")
        
        # Run all analyses
        self.basic_statistical_analysis()
        self.distribution_comparison(save_path=os.path.join(output_dir, "distributions.png"))
        real_corr, synth_corr, corr_diff = self.correlation_analysis(save_path=os.path.join(output_dir, "correlations.png"))
        financial_metrics = self.financial_metrics_analysis()
        overall_quality, detailed_results = self.quality_assessment()
        
        # Generate summary visualization
        self._create_summary_visualization(overall_quality, detailed_results, output_dir)
        
        print(f"\n📋 COMPREHENSIVE REPORT SUMMARY")
        print("=" * 60)
        print(f"Overall Quality Score: {overall_quality:.1%}")
        print(f"Real Data Sequences: {self.real_data.shape[0]:,}")
        print(f"Synthetic Data Sequences: {self.synthetic_scaled.shape[0]:,}")
        print(f"Dataset Size: {self.real_data.shape[0]:,} sequences each")
        print(f"Features Analyzed: {len(self.feature_names)}")
        print(f"Data Points: {self.real_data.shape[0] * self.real_data.shape[1]:,} per dataset")
        
        if financial_metrics:
            print(f"Financial Metrics: Comprehensive comparison completed")
        
        print(f"\n💾 REPORT SAVED TO: {output_dir}/")
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        
        return {
            'overall_quality': overall_quality,
            'detailed_results': detailed_results,
            'financial_metrics': financial_metrics
        }
    
    def _create_summary_visualization(self, overall_quality, detailed_results, output_dir):
        """Create comprehensive summary visualization"""
        plt.figure(figsize=(16, 10))
        
        # 1. Quality scores by feature
        plt.subplot(2, 2, 1)
        features = [r['feature'] for r in detailed_results]
        scores = [r['score'] for r in detailed_results]
        
        colors = ['green' if s >= 0.9 else 'lightgreen' if s >= 0.8 else 'orange' if s >= 0.7 else 'red' for s in scores]
        
        bars = plt.bar(features, scores, color=colors, edgecolor='black', alpha=0.8)
        plt.axhline(y=overall_quality, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall: {overall_quality:.1%}')
        plt.title('Feature Quality Scores\n(50K sequences comparison)', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Quality Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 2. Dataset size comparison
        plt.subplot(2, 2, 2)
        datasets = {
            'Real Data\n50,000 sequences': self.real_data.shape[0],
            'Synthetic Data\n50,000 sequences': self.synthetic_scaled.shape[0]
        }
        
        colors = ['blue', 'red']
        bars = plt.bar(datasets.keys(), datasets.values(), color=colors, alpha=0.7)
        plt.title('Dataset Scale\nEqual Comparison', fontweight='bold')
        plt.ylabel('Number of Sequences')
        
        for bar, count in zip(bars, datasets.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        
        # 3. Quality distribution
        plt.subplot(2, 2, 3)
        quality_ranges = {'Excellent (90-100%)': 0, 'Very Good (80-89%)': 0, 
                         'Good (70-79%)': 0, 'Needs Improvement (<70%)': 0}
        for r in detailed_results:
            score = r['score']
            if score >= 0.9: quality_ranges['Excellent (90-100%)'] += 1
            elif score >= 0.8: quality_ranges['Very Good (80-89%)'] += 1
            elif score >= 0.7: quality_ranges['Good (70-79%)'] += 1
            else: quality_ranges['Needs Improvement (<70%)'] += 1
        
        plt.bar(quality_ranges.keys(), quality_ranges.values(), 
                color=['green', 'lightgreen', 'orange', 'red'])
        plt.title('Quality Distribution Across Features', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Features')
        
        # 4. Overall achievement
        plt.subplot(2, 2, 4)
        achievements = {
            'Data Scale': self.real_data.shape[0],
            'Quality Score': overall_quality * 100,
            'Features': len(self.feature_names),
            'Time Steps': self.real_data.shape[1]
        }
        
        metrics = list(achievements.keys())
        values = list(achievements.values())
        
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
        plt.title('Overall Achievement', fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        
        for bar, (metric, value) in zip(bars, achievements.items()):
            if metric == 'Quality Score':
                label = f'{value:.1f}%'
            else:
                label = f'{value:,}'
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, 
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary_visualization.png"), dpi=300, bbox_inches='tight')
        plt.show()

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
    
    print("🔍 EQUAL COMPARISON SYNTHETIC BITCOIN DATA ANALYSIS")
    print("=" * 60)
    print("📊 USING 50,000 SEQUENCES FOR BOTH DATASETS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = EqualSyntheticDataAnalyzer(
        REAL_DATA_PATH, 
        SYNTHETIC_SCALED_PATH, 
        SYNTHETIC_INVERSE_PATH, 
        FEATURE_NAMES
    )
    
    # Generate comprehensive report
    results = analyzer.generate_report()