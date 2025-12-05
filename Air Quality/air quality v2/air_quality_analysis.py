# air_quality_analysis.py
"""
COMPREHENSIVE ANALYSIS OF AIR QUALITY TIME-GAN PIPELINE
Analyzes: Preprocessing → Training → Evaluation → Synthetic Data Quality
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import glob

class ComprehensiveAirQualityAnalyzer:
    """Analyze the complete air quality Time-GAN pipeline"""
    
    def __init__(self):
        self.results = {}
        self.analysis_dir = 'analysis_results'
        os.makedirs(self.analysis_dir, exist_ok=True)
        
    def analyze_preprocessing(self):
        """Analyze preprocessing step"""
        print("\n" + "="*60)
        print("1. PREPROCESSING ANALYSIS")
        print("="*60)
        
        analysis = {}
        
        # Check if preprocessing outputs exist
        data_dir = 'data/processed/air_quality'
        
        if not os.path.exists(data_dir):
            print(f"❌ Preprocessing directory not found: {data_dir}")
            return None
        
        # Load data
        try:
            train_data = np.load(os.path.join(data_dir, 'train.npy'))
            val_data = np.load(os.path.join(data_dir, 'val.npy'))
            test_data = np.load(os.path.join(data_dir, 'test.npy'))
            
            print(f"✓ Data loaded successfully")
            print(f"  Train: {train_data.shape}")
            print(f"  Validation: {val_data.shape}")
            print(f"  Test: {test_data.shape}")
            
            analysis['data_shapes'] = {
                'train': train_data.shape,
                'val': val_data.shape,
                'test': test_data.shape
            }
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Load metadata
        try:
            with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            print("\n📊 Preprocessing Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            analysis['metadata'] = metadata
            
        except:
            print("⚠️ Metadata not found")
        
        # Analyze data statistics
        print("\n📈 Data Statistics:")
        
        all_data = np.concatenate([train_data, val_data, test_data], axis=0)
        
        # Global statistics
        analysis['global_stats'] = {
            'min': float(all_data.min()),
            'max': float(all_data.max()),
            'mean': float(all_data.mean()),
            'std': float(all_data.std()),
            'median': float(np.median(all_data)),
            'skewness': float(stats.skew(all_data.flatten())),
            'kurtosis': float(stats.kurtosis(all_data.flatten()))
        }
        
        for stat, value in analysis['global_stats'].items():
            print(f"  {stat:15s}: {value:.4f}")
        
        # Feature-wise statistics
        n_features = train_data.shape[2]
        feature_stats = []
        
        for i in range(min(5, n_features)):  # Analyze first 5 features
            feature_data = all_data[:, :, i].flatten()
            
            stats_dict = {
                'feature': i,
                'mean': float(feature_data.mean()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'variance': float(feature_data.var())
            }
            feature_stats.append(stats_dict)
        
        analysis['feature_stats'] = feature_stats
        
        # Temporal analysis
        print("\n⏰ Temporal Analysis:")
        
        # Check autocorrelation
        def compute_temporal_stats(data):
            autocorrs = []
            cross_corrs = []
            
            for seq in data[:50]:  # Sample 50 sequences
                # Auto-correlation
                for f in range(min(3, data.shape[2])):
                    series = seq[:, f]
                    if len(series) > 3:
                        try:
                            ac = np.corrcoef(series[:-3], series[3:])[0, 1]
                            if not np.isnan(ac):
                                autocorrs.append(np.abs(ac))
                        except:
                            pass
                
                # Cross-correlation between first 2 features
                if data.shape[2] >= 2:
                    try:
                        cc = np.corrcoef(seq[:, 0], seq[:, 1])[0, 1]
                        if not np.isnan(cc):
                            cross_corrs.append(np.abs(cc))
                    except:
                        pass
            
            return {
                'mean_autocorr': np.mean(autocorrs) if autocorrs else 0,
                'mean_crosscorr': np.mean(cross_corrs) if cross_corrs else 0
            }
        
        temporal_stats = compute_temporal_stats(train_data)
        analysis['temporal_stats'] = temporal_stats
        
        print(f"  Mean Auto-correlation: {temporal_stats['mean_autocorr']:.4f}")
        print(f"  Mean Cross-correlation: {temporal_stats['mean_crosscorr']:.4f}")
        
        # Data quality assessment
        print("\n🔍 Data Quality Assessment:")
        
        # Check for NaNs
        nan_count = np.isnan(all_data).sum()
        analysis['nan_count'] = int(nan_count)
        print(f"  NaN values: {nan_count}")
        
        # Check for Infs
        inf_count = np.isinf(all_data).sum()
        analysis['inf_count'] = int(inf_count)
        print(f"  Infinite values: {inf_count}")
        
        # Check data range
        data_range = all_data.max() - all_data.min()
        analysis['data_range'] = float(data_range)
        print(f"  Data range: [{all_data.min():.3f}, {all_data.max():.3f}]")
        
        # Quality score
        quality_score = 1.0
        
        if nan_count > 0:
            quality_score *= 0.5
        
        if inf_count > 0:
            quality_score *= 0.5
        
        if data_range < 0.1:
            quality_score *= 0.7
        elif data_range > 10:
            quality_score *= 0.8
        
        analysis['preprocessing_quality_score'] = float(quality_score)
        
        if quality_score >= 0.9:
            print(f"  ✅ PREPROCESSING QUALITY: EXCELLENT ({quality_score:.2f})")
        elif quality_score >= 0.7:
            print(f"  ⚠️ PREPROCESSING QUALITY: GOOD ({quality_score:.2f})")
        elif quality_score >= 0.5:
            print(f"  ⚠️ PREPROCESSING QUALITY: FAIR ({quality_score:.2f})")
        else:
            print(f"  ❌ PREPROCESSING QUALITY: POOR ({quality_score:.2f})")
        
        self.results['preprocessing'] = analysis
        return analysis
    
    def analyze_training(self):
        """Analyze training process"""
        print("\n" + "="*60)
        print("2. TRAINING ANALYSIS")
        print("="*60)
        
        analysis = {}
        
        # Check for training checkpoints
        checkpoint_dirs = [
            'checkpoints/air_quality',
            'checkpoints/air_quality_fixed',
            'checkpoints/memory_safe'
        ]
        
        found_checkpoints = False
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                found_checkpoints = True
                analysis['checkpoint_dir'] = checkpoint_dir
                print(f"✓ Found checkpoints in: {checkpoint_dir}")
                break
        
        if not found_checkpoints:
            print("❌ No training checkpoints found!")
            return None
        
        # Look for training history files
        history_files = []
        for root, dirs, files in os.walk(analysis['checkpoint_dir']):
            for file in files:
                if 'history' in file.lower() or 'metrics' in file.lower():
                    history_files.append(os.path.join(root, file))
        
        # Look for JSON files with metrics
        json_files = glob.glob(os.path.join(analysis['checkpoint_dir'], '**/*.json'), recursive=True)
        
        # Try to load training history
        training_history = None
        best_metrics = None
        
        # First try to find checkpoint info
        checkpoint_info_files = [f for f in json_files if 'checkpoint' in f.lower() or 'best' in f.lower()]
        
        if checkpoint_info_files:
            try:
                with open(checkpoint_info_files[0], 'r') as f:
                    checkpoint_info = json.load(f)
                
                if 'metrics' in checkpoint_info:
                    best_metrics = checkpoint_info['metrics']
                    print(f"✓ Loaded best checkpoint metrics")
                    
                    print("\n🏆 Best Model Metrics:")
                    for key, value in best_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key:25s}: {value:.4f}")
                
                analysis['checkpoint_info'] = checkpoint_info
                
            except Exception as e:
                print(f"⚠️ Error loading checkpoint info: {e}")
        
        # Look for synthetic data
        synthetic_dirs = [
            'outputs/synthetic_air_quality',
            'outputs/synthetic_air_quality_fixed',
            'outputs/memory_safe'
        ]
        
        synthetic_data = None
        synthetic_path = None
        
        for synth_dir in synthetic_dirs:
            if os.path.exists(synth_dir):
                synth_files = glob.glob(os.path.join(synth_dir, 'synthetic_*.npy'))
                if synth_files:
                    synthetic_path = max(synth_files, key=os.path.getctime)
                    try:
                        synthetic_data = np.load(synthetic_path)
                        print(f"\n✓ Found synthetic data: {os.path.basename(synthetic_path)}")
                        print(f"  Shape: {synthetic_data.shape}")
                        analysis['synthetic_data'] = {
                            'path': synthetic_path,
                            'shape': synthetic_data.shape,
                            'samples': len(synthetic_data)
                        }
                        break
                    except Exception as e:
                        print(f"⚠️ Error loading synthetic data: {e}")
        
        if synthetic_data is None:
            print("\n❌ No synthetic data found!")
        else:
            # Analyze synthetic data quality
            print("\n📊 Synthetic Data Statistics:")
            synth_stats = {
                'min': float(synthetic_data.min()),
                'max': float(synthetic_data.max()),
                'mean': float(synthetic_data.mean()),
                'std': float(synthetic_data.std()),
                'range': float(synthetic_data.max() - synthetic_data.min())
            }
            
            for stat, value in synth_stats.items():
                print(f"  {stat:15s}: {value:.4f}")
            
            analysis['synthetic_stats'] = synth_stats
        
        # Training assessment
        print("\n🔍 Training Assessment:")
        
        training_score = 0.5  # Base score
        
        if best_metrics:
            if 'overall_score' in best_metrics:
                training_score = best_metrics['overall_score']
            elif 'mean_correlation' in best_metrics:
                # Estimate from mean correlation
                training_score = (best_metrics.get('mean_correlation', 0) + 1) / 2
        
        analysis['training_score'] = float(training_score)
        
        if training_score >= 0.7:
            print(f"  ✅ TRAINING QUALITY: EXCELLENT ({training_score:.2f})")
        elif training_score >= 0.5:
            print(f"  ⚠️ TRAINING QUALITY: GOOD ({training_score:.2f})")
        elif training_score >= 0.3:
            print(f"  ⚠️ TRAINING QUALITY: FAIR ({training_score:.2f})")
        else:
            print(f"  ❌ TRAINING QUALITY: POOR ({training_score:.2f})")
        
        # Issues detected
        issues = []
        
        if synthetic_data is None:
            issues.append("No synthetic data generated")
        
        if training_score < 0.4:
            issues.append("Low training score")
        
        if issues:
            print("\n⚠️ Issues detected:")
            for issue in issues:
                print(f"  • {issue}")
        
        analysis['issues'] = issues
        
        self.results['training'] = analysis
        return analysis
    
    def analyze_evaluation(self):
        """Analyze evaluation results"""
        print("\n" + "="*60)
        print("3. EVALUATION ANALYSIS")
        print("="*60)
        
        analysis = {}
        
        # Look for evaluation results
        eval_dirs = [
            'outputs/evaluation_results',
            'outputs/synthetic_air_quality_fixed',
            'outputs/memory_safe'
        ]
        
        eval_results = None
        eval_path = None
        
        for eval_dir in eval_dirs:
            if os.path.exists(eval_dir):
                # Look for JSON results
                json_files = glob.glob(os.path.join(eval_dir, '**/*.json'), recursive=True)
                
                # Prioritize files with 'results' or 'evaluation' in name
                result_files = [f for f in json_files if 'result' in f.lower() or 'eval' in f.lower()]
                
                if result_files:
                    eval_path = max(result_files, key=os.path.getctime)
                    try:
                        with open(eval_path, 'r') as f:
                            eval_results = json.load(f)
                        
                        print(f"✓ Found evaluation results: {os.path.basename(eval_path)}")
                        analysis['eval_path'] = eval_path
                        break
                    except Exception as e:
                        print(f"⚠️ Error loading evaluation results: {e}")
        
        if eval_results is None:
            print("❌ No evaluation results found!")
            
            # Try to compute evaluation from data
            return self._compute_evaluation_from_data()
        
        # Analyze evaluation results
        print("\n📊 Evaluation Results:")
        
        def print_metrics(metrics, prefix="  ", depth=0):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    print_metrics(value, prefix + "  ", depth + 1)
                elif isinstance(value, (int, float)):
                    print(f"{prefix}{key:30s}: {value:.4f}")
                else:
                    print(f"{prefix}{key:30s}: {value}")
        
        print_metrics(eval_results)
        
        # Extract overall score
        overall_score = None
        
        def find_overall_score(obj, path=""):
            nonlocal overall_score
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'overall' in key.lower() and 'score' in key.lower():
                        if isinstance(value, (int, float)):
                            overall_score = value
                            return True
                    if isinstance(value, dict):
                        if find_overall_score(value, f"{path}.{key}"):
                            return True
            return False
        
        find_overall_score(eval_results)
        
        if overall_score is not None:
            analysis['overall_score'] = float(overall_score)
            
            print(f"\n🏆 Overall Quality Score: {overall_score:.4f}")
            
            if overall_score >= 0.8:
                print("  ✅ SYNTHETIC DATA QUALITY: EXCELLENT")
                analysis['quality_rating'] = 'EXCELLENT'
            elif overall_score >= 0.6:
                print("  ⚠️ SYNTHETIC DATA QUALITY: GOOD")
                analysis['quality_rating'] = 'GOOD'
            elif overall_score >= 0.4:
                print("  ⚠️ SYNTHETIC DATA QUALITY: FAIR")
                analysis['quality_rating'] = 'FAIR'
            else:
                print("  ❌ SYNTHETIC DATA QUALITY: POOR")
                analysis['quality_rating'] = 'POOR'
        else:
            print("⚠️ Overall score not found in evaluation results")
        
        analysis['evaluation_results'] = eval_results
        
        # Check for visualizations
        viz_dirs = [
            'outputs/evaluation_plots',
            'evaluation_plots_simple'
        ]
        
        for viz_dir in viz_dirs:
            if os.path.exists(viz_dir):
                viz_files = os.listdir(viz_dir)
                if viz_files:
                    analysis['visualizations'] = {
                        'directory': viz_dir,
                        'files': viz_files[:5]  # First 5 files
                    }
                    print(f"\n📈 Visualizations found in: {viz_dir}")
                    print(f"  Files: {len(viz_files)}")
                    break
        
        self.results['evaluation'] = analysis
        return analysis
    
    def _compute_evaluation_from_data(self):
        """Compute evaluation metrics directly from data"""
        print("\n📊 Computing evaluation from data...")
        
        analysis = {}
        
        # Load real and synthetic data
        real_data_path = 'data/processed/air_quality/test.npy'
        
        if not os.path.exists(real_data_path):
            print(f"❌ Real data not found: {real_data_path}")
            return None
        
        # Find synthetic data
        synthetic_dirs = [
            'outputs/synthetic_air_quality',
            'outputs/synthetic_air_quality_fixed',
            'outputs/memory_safe'
        ]
        
        synthetic_path = None
        for synth_dir in synthetic_dirs:
            if os.path.exists(synth_dir):
                synth_files = glob.glob(os.path.join(synth_dir, 'synthetic_*.npy'))
                if synth_files:
                    synthetic_path = max(synth_files, key=os.path.getctime)
                    break
        
        if not synthetic_path:
            print("❌ No synthetic data found!")
            return None
        
        try:
            real_data = np.load(real_data_path)
            synthetic_data = np.load(synthetic_path)
            
            print(f"✓ Loaded data:")
            print(f"  Real: {real_data.shape}")
            print(f"  Synthetic: {synthetic_data.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Compute basic metrics
        metrics = self._compute_basic_comparison_metrics(real_data, synthetic_data)
        
        analysis['computed_metrics'] = metrics
        analysis['real_data_shape'] = real_data.shape
        analysis['synthetic_data_shape'] = synthetic_data.shape
        
        print("\n📊 Computed Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key:25s}: {value:.4f}")
        
        self.results['evaluation'] = analysis
        return analysis
    
    def _compute_basic_comparison_metrics(self, real_data, synthetic_data):
        """Compute basic comparison metrics"""
        metrics = {}
        
        # Use smaller samples for computation
        n_samples = min(100, len(real_data), len(synthetic_data))
        real_sample = real_data[:n_samples]
        synth_sample = synthetic_data[:n_samples]
        
        # Flatten data
        real_flat = real_sample.reshape(-1, real_sample.shape[2])
        synth_flat = synth_sample.reshape(-1, synth_sample.shape[2])
        
        # Basic statistics
        real_mean = np.mean(real_flat, axis=0)
        synth_mean = np.mean(synth_flat, axis=0)
        real_std = np.std(real_flat, axis=0)
        synth_std = np.std(synth_flat, axis=0)
        
        # Correlation coefficients
        try:
            mean_corr = np.corrcoef(real_mean, synth_mean)[0, 1]
            std_corr = np.corrcoef(real_std, synth_std)[0, 1]
        except:
            mean_corr = 0.0
            std_corr = 0.0
        
        metrics['mean_correlation'] = float(mean_corr) if not np.isnan(mean_corr) else 0.0
        metrics['std_correlation'] = float(std_corr) if not np.isnan(std_corr) else 0.0
        metrics['mean_mae'] = float(np.mean(np.abs(real_mean - synth_mean)))
        metrics['std_mae'] = float(np.mean(np.abs(real_std - synth_std)))
        
        # KS test for first 3 features
        ks_pvalues = []
        for f in range(min(3, real_flat.shape[1])):
            ks_stat, ks_p = stats.ks_2samp(real_flat[:500, f], synth_flat[:500, f])
            ks_pvalues.append(ks_p)
        
        metrics['ks_mean_pvalue'] = float(np.mean(ks_pvalues)) if ks_pvalues else 0.0
        
        # Temporal similarity
        def compute_autocorrelation_similarity(real_data, synth_data):
            real_ac = []
            synth_ac = []
            
            for i in range(min(20, len(real_data), len(synth_data))):
                for f in range(min(2, real_data.shape[2])):
                    real_series = real_data[i, :, f]
                    synth_series = synth_data[i, :, f]
                    
                    if len(real_series) > 3:
                        try:
                            r_ac = np.corrcoef(real_series[:-3], real_series[3:])[0, 1]
                            s_ac = np.corrcoef(synth_series[:-3], synth_series[3:])[0, 1]
                            
                            if not np.isnan(r_ac) and not np.isnan(s_ac):
                                real_ac.append(np.abs(r_ac))
                                synth_ac.append(np.abs(s_ac))
                        except:
                            pass
            
            if real_ac and synth_ac:
                similarity = 1 - np.abs(np.mean(real_ac) - np.mean(synth_ac))
                return max(0, min(1, similarity))
            
            return 0.5
        
        metrics['temporal_similarity'] = compute_autocorrelation_similarity(real_sample, synth_sample)
        
        # Overall score
        weights = {
            'mean_correlation': 0.3,
            'std_correlation': 0.2,
            'ks_mean_pvalue': 0.2,
            'temporal_similarity': 0.3
        }
        
        overall_score = 0
        for metric, weight in weights.items():
            score = metrics[metric]
            
            # Normalize scores
            if metric in ['mean_correlation', 'std_correlation']:
                score = (score + 1) / 2  # Convert from [-1,1] to [0,1]
            elif metric == 'ks_mean_pvalue':
                score = min(score * 5, 1.0)  # Scale p-value
            
            overall_score += score * weight
        
        metrics['overall_score'] = float(overall_score)
        
        return metrics
    
    def compare_real_vs_synthetic(self):
        """Detailed comparison between real and synthetic data"""
        print("\n" + "="*60)
        print("4. REAL vs SYNTHETIC COMPARISON")
        print("="*60)
        
        comparison = {}
        
        # Load data
        real_data_path = 'data/processed/air_quality/test.npy'
        
        if not os.path.exists(real_data_path):
            print(f"❌ Real data not found: {real_data_path}")
            return None
        
        # Find latest synthetic data
        synthetic_path = None
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.startswith('synthetic_') and file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    if synthetic_path is None or os.path.getctime(full_path) > os.path.getctime(synthetic_path):
                        synthetic_path = full_path
        
        if not synthetic_path:
            print("❌ No synthetic data found!")
            return None
        
        try:
            real_data = np.load(real_data_path)
            synthetic_data = np.load(synthetic_path)
            
            print(f"✓ Data loaded:")
            print(f"  Real: {real_data.shape}")
            print(f"  Synthetic: {synthetic_path}")
            print(f"            {synthetic_data.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Ensure same sample size
        n_samples = min(500, len(real_data), len(synthetic_data))
        real_data = real_data[:n_samples]
        synthetic_data = synthetic_data[:n_samples]
        
        comparison['sample_size'] = n_samples
        
        # 1. Statistical Comparison
        print("\n📊 Statistical Comparison:")
        
        real_stats = {
            'mean': np.mean(real_data, axis=(0, 1)),
            'std': np.std(real_data, axis=(0, 1)),
            'min': np.min(real_data, axis=(0, 1)),
            'max': np.max(real_data, axis=(0, 1))
        }
        
        synth_stats = {
            'mean': np.mean(synthetic_data, axis=(0, 1)),
            'std': np.std(synthetic_data, axis=(0, 1)),
            'min': np.min(synthetic_data, axis=(0, 1)),
            'max': np.max(synthetic_data, axis=(0, 1))
        }
        
        # Compute differences
        for stat in ['mean', 'std']:
            diff = np.abs(real_stats[stat] - synth_stats[stat])
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            
            print(f"\n  {stat.upper()} Differences:")
            print(f"    Average difference: {mean_diff:.4f}")
            print(f"    Maximum difference: {max_diff:.4f}")
            
            comparison[f'{stat}_difference'] = {
                'mean': float(mean_diff),
                'max': float(max_diff)
            }
        
        # 2. Distribution Comparison (KS Test)
        print("\n📈 Distribution Comparison (KS Test):")
        
        ks_results = []
        for f in range(min(5, real_data.shape[2])):
            real_flat = real_data[:, :, f].flatten()[:1000]
            synth_flat = synthetic_data[:, :, f].flatten()[:1000]
            
            ks_stat, ks_p = stats.ks_2samp(real_flat, synth_flat)
            ks_results.append({
                'feature': f,
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'significant': ks_p < 0.05
            })
            
            print(f"  Feature {f}: KS={ks_stat:.4f}, p={ks_p:.4f} {'❌' if ks_p < 0.05 else '✓'}")
        
        comparison['ks_test_results'] = ks_results
        
        # 3. Temporal Pattern Comparison
        print("\n⏰ Temporal Pattern Comparison:")
        
        def compute_temporal_metrics(data):
            metrics = {
                'autocorrelation': [],
                'cross_correlation': []
            }
            
            for seq in data[:50]:
                # Auto-correlation
                for f in range(min(3, data.shape[2])):
                    series = seq[:, f]
                    if len(series) > 3:
                        try:
                            ac = np.corrcoef(series[:-3], series[3:])[0, 1]
                            if not np.isnan(ac):
                                metrics['autocorrelation'].append(np.abs(ac))
                        except:
                            pass
                
                # Cross-correlation
                if data.shape[2] >= 2:
                    try:
                        cc = np.corrcoef(seq[:, 0], seq[:, 1])[0, 1]
                        if not np.isnan(cc):
                            metrics['cross_correlation'].append(np.abs(cc))
                    except:
                        pass
            
            # Compute means
            result = {}
            for key, values in metrics.items():
                result[key] = float(np.mean(values)) if values else 0.0
            
            return result
        
        real_temporal = compute_temporal_metrics(real_data)
        synth_temporal = compute_temporal_metrics(synthetic_data)
        
        print(f"  Real Auto-correlation:     {real_temporal['autocorrelation']:.4f}")
        print(f"  Synthetic Auto-correlation: {synth_temporal['autocorrelation']:.4f}")
        print(f"  Difference:                 {abs(real_temporal['autocorrelation'] - synth_temporal['autocorrelation']):.4f}")
        
        comparison['temporal_comparison'] = {
            'real': real_temporal,
            'synthetic': synth_temporal,
            'autocorr_diff': abs(real_temporal['autocorrelation'] - synth_temporal['autocorrelation'])
        }
        
        # 4. Feature Space Comparison (PCA)
        print("\n🎯 Feature Space Comparison (PCA):")
        
        # Flatten and sample
        real_flat = real_data.reshape(-1, real_data.shape[2])
        synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[2])
        
        n_samples_pca = min(1000, len(real_flat), len(synth_flat))
        real_sample = real_flat[:n_samples_pca]
        synth_sample = synth_flat[:n_samples_pca]
        
        combined = np.vstack([real_sample, synth_sample])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        real_pca = pca_result[:n_samples_pca]
        synth_pca = pca_result[n_samples_pca:]
        
        # Compute centroids and distances
        real_centroid = np.mean(real_pca, axis=0)
        synth_centroid = np.mean(synth_pca, axis=0)
        centroid_distance = np.linalg.norm(real_centroid - synth_centroid)
        
        print(f"  Centroid distance: {centroid_distance:.4f}")
        print(f"  Variance explained: {np.sum(pca.explained_variance_ratio_):.2%}")
        
        comparison['pca_analysis'] = {
            'centroid_distance': float(centroid_distance),
            'variance_explained': float(np.sum(pca.explained_variance_ratio_)),
            'real_centroid': real_centroid.tolist(),
            'synth_centroid': synth_centroid.tolist()
        }
        
        # 5. Overall Similarity Score
        print("\n🏆 Overall Similarity Assessment:")
        
        # Compute component scores
        component_scores = {}
        
        # Statistical similarity (40%)
        stat_similarity = 1 - comparison['mean_difference']['mean']
        component_scores['statistical'] = max(0, min(1, stat_similarity))
        
        # Distribution similarity (30%)
        significant_ks = sum(1 for r in ks_results if r['significant'])
        dist_similarity = 1 - (significant_ks / len(ks_results))
        component_scores['distribution'] = dist_similarity
        
        # Temporal similarity (20%)
        temp_similarity = 1 - comparison['temporal_comparison']['autocorr_diff']
        component_scores['temporal'] = max(0, min(1, temp_similarity))
        
        # Feature space similarity (10%)
        space_similarity = 1 - min(centroid_distance / 5, 1)
        component_scores['feature_space'] = space_similarity
        
        # Weighted overall score
        weights = {
            'statistical': 0.4,
            'distribution': 0.3,
            'temporal': 0.2,
            'feature_space': 0.1
        }
        
        overall_similarity = 0
        for component, score in component_scores.items():
            overall_similarity += score * weights[component]
        
        comparison['similarity_scores'] = component_scores
        comparison['overall_similarity'] = float(overall_similarity)
        
        print(f"\n  Component Scores:")
        for component, score in component_scores.items():
            print(f"    {component:15s}: {score:.3f}")
        
        print(f"\n  Overall Similarity: {overall_similarity:.3f}")
        
        if overall_similarity >= 0.8:
            print(f"  ✅ EXCELLENT similarity with real data!")
        elif overall_similarity >= 0.6:
            print(f"  ⚠️ GOOD similarity with real data")
        elif overall_similarity >= 0.4:
            print(f"  ⚠️ FAIR similarity with real data")
        else:
            print(f"  ❌ POOR similarity with real data")
        
        self.results['comparison'] = comparison
        return comparison
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("5. GENERATING VISUALIZATIONS")
        print("="*60)
        
        viz_dir = os.path.join(self.analysis_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load data if available
        try:
            real_data = np.load('data/processed/air_quality/test.npy')
            
            # Find synthetic data
            synthetic_path = None
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.startswith('synthetic_') and file.endswith('.npy'):
                        full_path = os.path.join(root, file)
                        if synthetic_path is None or os.path.getctime(full_path) > os.path.getctime(synthetic_path):
                            synthetic_path = full_path
            
            if synthetic_path:
                synthetic_data = np.load(synthetic_path)
                
                print(f"✓ Data loaded for visualizations")
                print(f"  Real: {real_data.shape}")
                print(f"  Synthetic: {synthetic_data.shape}")
                
                # Create visualizations
                self._create_comparison_plots(real_data, synthetic_data, viz_dir)
                
            else:
                print("⚠️ No synthetic data found for visualizations")
                
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
        
        print(f"\n📊 Visualizations will be saved to: {viz_dir}")
    
    def _create_comparison_plots(self, real_data, synthetic_data, output_dir):
        """Create comparison plots"""
        # Use smaller samples for plotting
        n_samples = min(100, len(real_data), len(synthetic_data))
        real_sample = real_data[:n_samples]
        synth_sample = synthetic_data[:n_samples]
        
        # 1. Statistical Summary Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Means comparison
        real_means = np.mean(real_sample, axis=(0, 1))
        synth_means = np.mean(synth_sample, axis=(0, 1))
        
        axes[0, 0].scatter(real_means, synth_means, alpha=0.6)
        axes[0, 0].plot([real_means.min(), real_means.max()], 
                       [real_means.min(), real_means.max()], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Real Means')
        axes[0, 0].set_ylabel('Synthetic Means')
        axes[0, 0].set_title('Feature Means Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Stds comparison
        real_stds = np.std(real_sample, axis=(0, 1))
        synth_stds = np.std(synth_sample, axis=(0, 1))
        
        axes[0, 1].scatter(real_stds, synth_stds, alpha=0.6)
        axes[0, 1].plot([real_stds.min(), real_stds.max()], 
                       [real_stds.min(), real_stds.max()], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Real Stds')
        axes[0, 1].set_ylabel('Synthetic Stds')
        axes[0, 1].set_title('Feature Standard Deviations Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation matrices
        real_corr = np.corrcoef(real_sample.reshape(-1, real_sample.shape[2]).T)
        synth_corr = np.corrcoef(synth_sample.reshape(-1, synth_sample.shape[2]).T)
        
        vmin = min(real_corr.min(), synth_corr.min())
        vmax = max(real_corr.max(), synth_corr.max())
        
        im1 = axes[1, 0].imshow(real_corr, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Real Data Correlation Matrix')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(synth_corr, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Synthetic Data Correlation Matrix')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statistical_comparison.png'), dpi=150)
        plt.close()
        
        # 2. Distribution Comparison
        n_features = min(6, real_sample.shape[2])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            real_flat = real_sample[:, :, i].flatten()[:1000]
            synth_flat = synth_sample[:, :, i].flatten()[:1000]
            
            ax.hist(real_flat, bins=50, alpha=0.5, density=True, label='Real')
            ax.hist(synth_flat, bins=50, alpha=0.5, density=True, label='Synthetic')
            ax.set_title(f'Feature {i} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=150)
        plt.close()
        
        # 3. Time Series Comparison
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(min(6, len(axes))):
            ax = axes[i]
            
            # Plot first feature
            real_series = real_sample[i, :, 0]
            synth_series = synth_sample[i, :, 0]
            
            ax.plot(real_series, 'b-', alpha=0.8, linewidth=2, label='Real')
            ax.plot(synth_series, 'r--', alpha=0.8, linewidth=2, label='Synthetic')
            
            ax.set_title(f'Sample {i} - Temporal Pattern')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'), dpi=150)
        plt.close()
        
        print(f"✓ Created 3 visualization plots")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("6. GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_path = os.path.join(self.analysis_dir, 'comprehensive_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE AIR QUALITY TIME-GAN PIPELINE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            # 1. Executive Summary
            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n\n")
            
            overall_status = "✅ COMPLETE" if all(k in self.results for k in ['preprocessing', 'training', 'evaluation']) else "⚠️ PARTIAL"
            f.write(f"Pipeline Status: {overall_status}\n\n")
            
            # 2. Preprocessing Summary
            if 'preprocessing' in self.results:
                prep = self.results['preprocessing']
                f.write("2. PREPROCESSING ANALYSIS\n")
                f.write("-"*40 + "\n\n")
                
                if 'data_shapes' in prep:
                    shapes = prep['data_shapes']
                    f.write(f"Data Shapes:\n")
                    f.write(f"  Training:      {shapes['train']}\n")
                    f.write(f"  Validation:    {shapes['val']}\n")
                    f.write(f"  Test:          {shapes['test']}\n\n")
                
                if 'global_stats' in prep:
                    f.write(f"Global Statistics:\n")
                    for stat, value in prep['global_stats'].items():
                        f.write(f"  {stat:15s}: {value:.4f}\n")
                    f.write("\n")
                
                if 'preprocessing_quality_score' in prep:
                    score = prep['preprocessing_quality_score']
                    f.write(f"Preprocessing Quality Score: {score:.3f}\n")
                    if score >= 0.9:
                        f.write("  Rating: EXCELLENT\n")
                    elif score >= 0.7:
                        f.write("  Rating: GOOD\n")
                    elif score >= 0.5:
                        f.write("  Rating: FAIR\n")
                    else:
                        f.write("  Rating: POOR\n")
                f.write("\n")
            
            # 3. Training Summary
            if 'training' in self.results:
                train = self.results['training']
                f.write("3. TRAINING ANALYSIS\n")
                f.write("-"*40 + "\n\n")
                
                if 'checkpoint_dir' in train:
                    f.write(f"Checkpoint Directory: {train['checkpoint_dir']}\n")
                
                if 'synthetic_data' in train:
                    synth = train['synthetic_data']
                    f.write(f"\nSynthetic Data Generated:\n")
                    f.write(f"  File: {os.path.basename(synth['path'])}\n")
                    f.write(f"  Shape: {synth['shape']}\n")
                    f.write(f"  Samples: {synth['samples']:,}\n\n")
                
                if 'training_score' in train:
                    score = train['training_score']
                    f.write(f"Training Quality Score: {score:.3f}\n")
                    if score >= 0.7:
                        f.write("  Rating: EXCELLENT\n")
                    elif score >= 0.5:
                        f.write("  Rating: GOOD\n")
                    elif score >= 0.3:
                        f.write("  Rating: FAIR\n")
                    else:
                        f.write("  Rating: POOR\n")
                
                if 'issues' in train and train['issues']:
                    f.write(f"\n⚠️ Issues Detected:\n")
                    for issue in train['issues']:
                        f.write(f"  • {issue}\n")
                f.write("\n")
            
            # 4. Evaluation Summary
            if 'evaluation' in self.results:
                eval_res = self.results['evaluation']
                f.write("4. EVALUATION ANALYSIS\n")
                f.write("-"*40 + "\n\n")
                
                if 'overall_score' in eval_res:
                    score = eval_res['overall_score']
                    f.write(f"Overall Synthetic Data Quality Score: {score:.3f}\n")
                    
                    if score >= 0.8:
                        f.write("  Rating: EXCELLENT - Highly realistic synthetic data\n")
                    elif score >= 0.6:
                        f.write("  Rating: GOOD - Realistic synthetic data\n")
                    elif score >= 0.4:
                        f.write("  Rating: FAIR - Acceptable with some issues\n")
                    else:
                        f.write("  Rating: POOR - Needs significant improvement\n")
                
                if 'computed_metrics' in eval_res:
                    f.write(f"\nComputed Metrics:\n")
                    metrics = eval_res['computed_metrics']
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key:25s}: {value:.4f}\n")
                f.write("\n")
            
            # 5. Comparison Summary
            if 'comparison' in self.results:
                comp = self.results['comparison']
                f.write("5. REAL vs SYNTHETIC COMPARISON\n")
                f.write("-"*40 + "\n\n")
                
                if 'overall_similarity' in comp:
                    similarity = comp['overall_similarity']
                    f.write(f"Overall Similarity Score: {similarity:.3f}\n\n")
                    
                    f.write(f"Component Scores:\n")
                    if 'similarity_scores' in comp:
                        for component, score in comp['similarity_scores'].items():
                            f.write(f"  {component:20s}: {score:.3f}\n")
                    f.write("\n")
            
            # 6. Recommendations
            f.write("6. RECOMMENDATIONS\n")
            f.write("-"*40 + "\n\n")
            
            recommendations = []
            
            # Check preprocessing
            if 'preprocessing' in self.results:
                prep = self.results['preprocessing']
                if prep.get('preprocessing_quality_score', 0) < 0.7:
                    recommendations.append("Improve data preprocessing - check for outliers and scaling")
                if prep.get('nan_count', 0) > 0 or prep.get('inf_count', 0) > 0:
                    recommendations.append("Clean data by removing/replacing NaN and infinite values")
            
            # Check training
            if 'training' in self.results:
                train = self.results['training']
                if train.get('training_score', 0) < 0.5:
                    recommendations.append("Increase training epochs or adjust model architecture")
                if train.get('issues'):
                    for issue in train['issues']:
                        recommendations.append(f"Address: {issue}")
            
            # Check evaluation
            if 'evaluation' in self.results:
                eval_res = self.results['evaluation']
                if eval_res.get('overall_score', 0) < 0.6:
                    recommendations.append("Improve TimeGAN training for better synthetic data quality")
                if 'quality_rating' in eval_res and eval_res['quality_rating'] in ['FAIR', 'POOR']:
                    recommendations.append("Consider adjusting TimeGAN hyperparameters or architecture")
            
            # Check comparison
            if 'comparison' in self.results:
                comp = self.results['comparison']
                if comp.get('overall_similarity', 0) < 0.6:
                    recommendations.append("Focus on improving temporal pattern generation")
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("Pipeline is working well. Consider generating more synthetic data")
                recommendations.append("Experiment with different sequence lengths and features")
                recommendations.append("Try different TimeGAN architectures for comparison")
            
            # Write recommendations
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Comprehensive report generated: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("COMPREHENSIVE AIR QUALITY TIME-GAN PIPELINE ANALYSIS")
        print("="*80)
        
        start_time = datetime.now()
        
        # Run all analyses
        print("\n🔍 Running analyses...")
        
        self.analyze_preprocessing()
        self.analyze_training()
        self.analyze_evaluation()
        self.compare_real_vs_synthetic()
        
        # Generate outputs
        print("\n📊 Generating outputs...")
        
        self.generate_visualizations()
        report_path = self.generate_summary_report()
        
        # Calculate execution time
        exec_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📁 Analysis results saved to: {self.analysis_dir}")
        print(f"📄 Comprehensive report: {report_path}")
        print(f"⏱️  Execution time: {exec_time:.1f} seconds")
        
        # Final summary
        print("\n🏁 FINAL SUMMARY:")
        
        scores = {}
        
        if 'preprocessing' in self.results:
            prep_score = self.results['preprocessing'].get('preprocessing_quality_score', 0)
            scores['Preprocessing'] = prep_score
            print(f"  Preprocessing:       {prep_score:.3f}")
        
        if 'training' in self.results:
            train_score = self.results['training'].get('training_score', 0)
            scores['Training'] = train_score
            print(f"  Training:            {train_score:.3f}")
        
        if 'evaluation' in self.results:
            eval_score = self.results['evaluation'].get('overall_score', 0)
            scores['Evaluation'] = eval_score
            print(f"  Evaluation:          {eval_score:.3f}")
        
        if 'comparison' in self.results:
            comp_score = self.results['comparison'].get('overall_similarity', 0)
            scores['Comparison'] = comp_score
            print(f"  Real vs Synthetic:   {comp_score:.3f}")
        
        if scores:
            avg_score = np.mean(list(scores.values()))
            print(f"\n  Average Score:       {avg_score:.3f}")
            
            if avg_score >= 0.7:
                print("\n🎉 OVERALL STATUS: EXCELLENT - Pipeline is working very well!")
            elif avg_score >= 0.5:
                print("\n✅ OVERALL STATUS: GOOD - Pipeline is functional")
            elif avg_score >= 0.3:
                print("\n⚠️ OVERALL STATUS: FAIR - Some improvements needed")
            else:
                print("\n❌ OVERALL STATUS: POOR - Significant improvements needed")
        
        return self.results

def main():
    """Main analysis function"""
    analyzer = ComprehensiveAirQualityAnalyzer()
    results = analyzer.run_complete_analysis()
    
    # Save full results as JSON
    results_path = os.path.join(analyzer.analysis_dir, 'analysis_results.json')
    
    # Convert numpy objects to serializable format
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
    
    serializable_results = convert_for_json(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📋 Full results saved as JSON: {results_path}")
    
    return results

if __name__ == "__main__":
    main()