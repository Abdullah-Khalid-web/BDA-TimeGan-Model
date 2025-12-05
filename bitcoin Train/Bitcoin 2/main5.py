# # import os
# # import pickle
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import StandardScaler
# # import json

# # def generate_plots_from_existing_data():
# #     """Generate plots from your already trained model and data"""
    
# #     print("📊 Generating plots from existing data...")
    
# #     try:
# #         # Load data
# #         with open('outputs/processed/scaler.pkl', 'rb') as f:
# #             scaler = pickle.load(f)
        
# #         test_data = np.load('outputs/processed/test.npy')
# #         synthetic_data = np.load('outputs/synthetic/synthetic_bitcoin.npy')
        
# #         # Load feature names
# #         with open('outputs/processed/features.txt', 'r') as f:
# #             feature_names = [line.strip() for line in f]
        
# #         # Try to load training history (optional)
# #         history = {}
# #         try:
# #             with open('outputs/synthetic/training_history.json', 'r') as f:
# #                 history = json.load(f)
# #             print("✅ Loaded training history")
# #         except FileNotFoundError:
# #             print("⚠️ Training history not found, continuing without it...")
# #             history = {}
        
# #         # Create comprehensive visualization
# #         create_comprehensive_plots(test_data, synthetic_data, history, feature_names, scaler)
        
# #     except FileNotFoundError as e:
# #         print(f"❌ Missing required file: {e}")
# #         print("Please run the TimeGAN training first or check file paths.")
# #         return False
    
# #     return True

# # def create_comprehensive_plots(real_data, synthetic_data, history, feature_names, scaler):
# #     """Create detailed comparison plots"""
    
# #     # Inverse transform the data for plotting
# #     print("🔄 Inverse transforming data for plotting...")
# #     n, T, D = real_data.shape
# #     real_data_denorm = scaler.inverse_transform(real_data.reshape(-1, D)).reshape(n, T, D)
    
# #     n_synth, T_synth, D_synth = synthetic_data.shape
# #     synthetic_data_denorm = scaler.inverse_transform(synthetic_data.reshape(-1, D_synth)).reshape(n_synth, T_synth, D_synth)
    
# #     fig, axes = plt.subplots(3, 3, figsize=(20, 15))
# #     fig.suptitle('TimeGAN Synthetic Bitcoin Data Analysis', fontsize=16, fontweight='bold')
    
# #     # 1. Training Loss History (if available)
# #     if history:
# #         axes[0,0].plot(history.get('embedder_loss', []), label='Embedder', alpha=0.7)
# #         axes[0,0].plot(history.get('supervisor_loss', []), label='Supervisor', alpha=0.7)
# #         axes[0,0].plot(history.get('discriminator_loss', []), label='Discriminator', alpha=0.7)
# #         axes[0,0].plot(history.get('generator_loss', []), label='Generator', alpha=0.7)
# #         axes[0,0].set_title('Training Loss History')
# #         axes[0,0].set_xlabel('Epoch')
# #         axes[0,0].set_ylabel('Loss')
# #         axes[0,0].legend()
# #         axes[0,0].grid(True, alpha=0.3)
# #     else:
# #         axes[0,0].text(0.5, 0.5, 'Training History\nNot Available', 
# #                       ha='center', va='center', transform=axes[0,0].transAxes, fontsize=12)
# #         axes[0,0].set_title('Training Loss History')
    
# #     # 2. Sample Comparison (First Feature)
# #     n_show = min(8, real_data_denorm.shape[0])
# #     time_steps = range(real_data_denorm.shape[1])
    
# #     for i in range(n_show):
# #         alpha = 0.7 if i < 3 else 0.3
# #         lw = 2 if i < 3 else 1
# #         if i == 0:
# #             axes[0,1].plot(time_steps, real_data_denorm[i, :, 0], 'g-', alpha=alpha, linewidth=lw, label='Real')
# #             axes[0,1].plot(time_steps, synthetic_data_denorm[i, :, 0], 'orange', alpha=alpha, linewidth=lw, label='Synthetic')
# #         else:
# #             axes[0,1].plot(time_steps, real_data_denorm[i, :, 0], 'g-', alpha=alpha, linewidth=lw)
# #             axes[0,1].plot(time_steps, synthetic_data_denorm[i, :, 0], 'orange', alpha=alpha, linewidth=lw)
# #     axes[0,1].set_title(f'Sample Comparison - {feature_names[0]}')
# #     axes[0,1].set_xlabel('Time Steps')
# #     axes[0,1].set_ylabel('Value')
# #     axes[0,1].legend()
# #     axes[0,1].grid(True, alpha=0.3)
    
# #     # 3. Distribution Comparison
# #     real_flat = real_data_denorm.reshape(-1, real_data_denorm.shape[-1])
# #     synth_flat = synthetic_data_denorm.reshape(-1, synthetic_data_denorm.shape[-1])
    
# #     axes[0,2].hist(real_flat[:, 0], bins=50, alpha=0.6, label='Real', density=True, color='green')
# #     axes[0,2].hist(synth_flat[:, 0], bins=50, alpha=0.6, label='Synthetic', density=True, color='orange')
# #     axes[0,2].set_title('Distribution Comparison')
# #     axes[0,2].set_xlabel('Value')
# #     axes[0,2].set_ylabel('Density')
# #     axes[0,2].legend()
# #     axes[0,2].grid(True, alpha=0.3)
    
# #     # 4. Correlation Matrix - Real Data
# #     if len(feature_names) >= 5:
# #         n_features_show = 5
# #     else:
# #         n_features_show = len(feature_names)
    
# #     real_corr = np.corrcoef(real_flat[:, :n_features_show].T)
# #     im1 = axes[1,0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
# #     axes[1,0].set_title('Real Data Correlations')
# #     axes[1,0].set_xticks(range(n_features_show))
# #     axes[1,0].set_xticklabels(feature_names[:n_features_show], rotation=45)
# #     axes[1,0].set_yticks(range(n_features_show))
# #     axes[1,0].set_yticklabels(feature_names[:n_features_show])
# #     plt.colorbar(im1, ax=axes[1,0])
    
# #     # 5. Correlation Matrix - Synthetic Data
# #     synth_corr = np.corrcoef(synth_flat[:, :n_features_show].T)
# #     im2 = axes[1,1].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1)
# #     axes[1,1].set_title('Synthetic Data Correlations')
# #     axes[1,1].set_xticks(range(n_features_show))
# #     axes[1,1].set_xticklabels(feature_names[:n_features_show], rotation=45)
# #     axes[1,1].set_yticks(range(n_features_show))
# #     axes[1,1].set_yticklabels(feature_names[:n_features_show])
# #     plt.colorbar(im2, ax=axes[1,1])
    
# #     # 6. Correlation Differences
# #     corr_diff = np.abs(real_corr - synth_corr)
# #     im3 = axes[1,2].imshow(corr_diff, cmap='hot', vmin=0, vmax=0.5)
# #     axes[1,2].set_title('Correlation Differences')
# #     axes[1,2].set_xticks(range(n_features_show))
# #     axes[1,2].set_xticklabels(feature_names[:n_features_show], rotation=45)
# #     axes[1,2].set_yticks(range(n_features_show))
# #     axes[1,2].set_yticklabels(feature_names[:n_features_show])
# #     plt.colorbar(im3, ax=axes[1,2])
    
# #     # 7. Temporal Patterns - Mean & Std
# #     real_mean = np.mean(real_data_denorm[:, :, 0], axis=0)
# #     real_std = np.std(real_data_denorm[:, :, 0], axis=0)
# #     synth_mean = np.mean(synthetic_data_denorm[:, :, 0], axis=0)
# #     synth_std = np.std(synthetic_data_denorm[:, :, 0], axis=0)
    
# #     axes[2,0].plot(real_mean, 'g-', label='Real Mean', linewidth=2)
# #     axes[2,0].fill_between(range(len(real_mean)), real_mean - real_std, real_mean + real_std, 
# #                           alpha=0.3, color='green', label='Real Std')
# #     axes[2,0].plot(synth_mean, 'orange', label='Synthetic Mean', linewidth=2)
# #     axes[2,0].fill_between(range(len(synth_mean)), synth_mean - synth_std, synth_mean + synth_std, 
# #                           alpha=0.3, color='orange', label='Synthetic Std')
# #     axes[2,0].set_title('Temporal Pattern Comparison')
# #     axes[2,0].set_xlabel('Time Steps')
# #     axes[2,0].set_ylabel(feature_names[0])
# #     axes[2,0].legend()
# #     axes[2,0].grid(True, alpha=0.3)
    
# #     # 8. Feature Relationships
# #     if len(feature_names) >= 2:
# #         axes[2,1].scatter(real_flat[:2000, 0], real_flat[:2000, 1], alpha=0.5, 
# #                          s=10, color='green', label='Real')
# #         axes[2,1].scatter(synth_flat[:2000, 0], synth_flat[:2000, 1], alpha=0.5, 
# #                          s=10, color='orange', label='Synthetic')
# #         axes[2,1].set_xlabel(feature_names[0])
# #         axes[2,1].set_ylabel(feature_names[1])
# #         axes[2,1].set_title('Feature Relationship')
# #         axes[2,1].legend()
# #         axes[2,1].grid(True, alpha=0.3)
# #     else:
# #         axes[2,1].text(0.5, 0.5, 'Feature Relationship\nNot Available\n(Need at least 2 features)', 
# #                       ha='center', va='center', transform=axes[2,1].transAxes, fontsize=12)
# #         axes[2,1].set_title('Feature Relationship')
    
# #     # 9. Data Quality Metrics
# #     # Calculate some basic quality metrics
# #     mean_diff = np.mean(np.abs(np.mean(real_flat, axis=0) - np.mean(synth_flat, axis=0)))
# #     std_diff = np.mean(np.abs(np.std(real_flat, axis=0) - np.std(synth_flat, axis=0)))
    
# #     axes[2,2].bar(['Mean Diff', 'Std Diff'], [mean_diff, std_diff], color=['skyblue', 'lightcoral'])
# #     axes[2,2].set_title('Data Quality Metrics')
# #     axes[2,2].set_ylabel('Difference')
# #     axes[2,2].grid(True, alpha=0.3)
    
# #     # Add value labels on bars
# #     for i, v in enumerate([mean_diff, std_diff]):
# #         axes[2,2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
# #     plt.tight_layout()
# #     plt.savefig('outputs/comprehensive_timegan_analysis.png', dpi=300, bbox_inches='tight')
# #     plt.show()
    
# #     print("💾 Plot saved as: outputs/comprehensive_timegan_analysis.png")
# #     print("✅ Plot generation completed successfully!")

# # # Run this to generate plots
# # if __name__ == "__main__":
# #     generate_plots_from_existing_data()




# # import numpy as np
# # import matplotlib.pyplot as plt
# # import pickle

# # def quick_plot():
# #     """Quick plot with available data"""
# #     try:
# #         # Load synthetic data
# #         synthetic_data = np.load('outputs/synthetic/synthetic_bitcoin.npy')
        
# #         # Load scaler and test data if available
# #         try:
# #             with open('outputs/processed/scaler.pkl', 'rb') as f:
# #                 scaler = pickle.load(f)
# #             test_data = np.load('outputs/processed/test.npy')
            
# #             # Inverse transform
# #             n, T, D = test_data.shape
# #             test_data_denorm = scaler.inverse_transform(test_data.reshape(-1, D)).reshape(n, T, D)
# #         except:
# #             print("⚠️ Using synthetic data only")
# #             test_data_denorm = None
        
# #         # Create simple plot
# #         plt.figure(figsize=(15, 10))
        
# #         if test_data_denorm is not None:
# #             # Plot first 5 samples from both datasets
# #             for i in range(min(5, test_data_denorm.shape[0])):
# #                 plt.subplot(2, 1, 1)
# #                 plt.plot(test_data_denorm[i, :, 0], 'g-', alpha=0.7, label='Real' if i == 0 else "")
# #                 plt.subplot(2, 1, 2)
# #                 plt.plot(synthetic_data[i, :, 0], 'orange', alpha=0.7, label='Synthetic' if i == 0 else "")
            
# #             plt.subplot(2, 1, 1)
# #             plt.title('Real Data Samples')
# #             plt.legend()
# #             plt.grid(True, alpha=0.3)
            
# #             plt.subplot(2, 1, 2)
# #             plt.title('Synthetic Data Samples')
# #             plt.legend()
# #             plt.grid(True, alpha=0.3)
# #         else:
# #             # Just plot synthetic data
# #             for i in range(min(10, synthetic_data.shape[0])):
# #                 plt.plot(synthetic_data[i, :, 0], alpha=0.7)
# #             plt.title('Synthetic Data Samples')
# #             plt.grid(True, alpha=0.3)
        
# #         plt.tight_layout()
# #         plt.savefig('outputs/quick_synthetic_plot.png', dpi=300, bbox_inches='tight')
# #         plt.show()
        
# #         print(f"📊 Synthetic data shape: {synthetic_data.shape}")
# #         print("💾 Quick plot saved as: outputs/quick_synthetic_plot.png")
        
# #     except Exception as e:
# #         print(f"❌ Error: {e}")
# #         print("Please make sure synthetic data exists at: outputs/synthetic/synthetic_bitcoin.npy")

# # quick_plot()










# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import json

# def generate_plots_from_existing_data():
#     """Generate plots from your already trained model and data"""
    
#     print("📊 Generating plots from existing data...")
    
#     try:
#         # Load data
#         with open('outputs/processed/scaler.pkl', 'rb') as f:
#             scaler = pickle.load(f)
        
#         test_data = np.load('outputs/processed/test.npy')
#         synthetic_data = np.load('outputs/synthetic/synthetic_bitcoin.npy')
        
#         # Load feature names
#         with open('outputs/processed/features.txt', 'r') as f:
#             feature_names = [line.strip() for line in f]
        
#         print(f"✅ Loaded data:")
#         print(f"   Test data shape: {test_data.shape}")
#         print(f"   Synthetic data shape: {synthetic_data.shape}")
#         print(f"   Features: {feature_names}")
        
#         # Try to load training history (optional)
#         history = {}
#         try:
#             with open('outputs/synthetic/training_history.json', 'r') as f:
#                 history = json.load(f)
#             print("✅ Loaded training history")
#         except FileNotFoundError:
#             print("⚠️ Training history not found, continuing without it...")
#             history = {}
        
#         # Create comprehensive visualization
#         create_comprehensive_plots(test_data, synthetic_data, history, feature_names, scaler)
        
#     except FileNotFoundError as e:
#         print(f"❌ Missing required file: {e}")
#         print("Please run the TimeGAN training first or check file paths.")
#         return False
    
#     return True

# def create_comprehensive_plots(real_data_scaled, synthetic_data_scaled, history, feature_names, scaler):
#     """Create detailed comparison plots with proper inverse scaling"""
    
#     print("🔄 Applying inverse transformation...")
    
#     # Reshape data for inverse transform
#     n_real, T_real, D_real = real_data_scaled.shape
#     n_synth, T_synth, D_synth = synthetic_data_scaled.shape
    
#     # Inverse transform both datasets
#     real_data = scaler.inverse_transform(
#         real_data_scaled.reshape(-1, D_real)
#     ).reshape(n_real, T_real, D_real)
    
#     synthetic_data = scaler.inverse_transform(
#         synthetic_data_scaled.reshape(-1, D_synth)
#     ).reshape(n_synth, T_synth, D_synth)
    
#     print(f"📊 After inverse transform:")
#     print(f"   Real data range: [{np.min(real_data):.2f}, {np.max(real_data):.2f}]")
#     print(f"   Synthetic data range: [{np.min(synthetic_data):.2f}, {np.max(synthetic_data):.2f}]")
    
#     # Create the figure
#     fig, axes = plt.subplots(3, 3, figsize=(20, 15))
#     fig.suptitle('TimeGAN Synthetic Bitcoin Data Analysis', fontsize=16, fontweight='bold')
    
#     # Plot 1: Training History (if available)
#     if history and any(len(history.get(key, [])) > 0 for key in history):
#         epochs = range(len(history.get('discriminator_loss', [])))
        
#         if history.get('embedder_loss'):
#             axes[0,0].plot(epochs, history['embedder_loss'], label='Embedder', alpha=0.7, linewidth=1)
#         if history.get('supervisor_loss'):
#             axes[0,0].plot(epochs, history['supervisor_loss'], label='Supervisor', alpha=0.7, linewidth=1)
#         if history.get('discriminator_loss'):
#             axes[0,0].plot(epochs, history['discriminator_loss'], label='Discriminator', alpha=0.7, linewidth=1)
#         if history.get('generator_loss'):
#             axes[0,0].plot(epochs, history['generator_loss'], label='Generator', alpha=0.7, linewidth=1)
        
#         axes[0,0].set_title('Training Loss History')
#         axes[0,0].set_xlabel('Epoch')
#         axes[0,0].set_ylabel('Loss')
#         axes[0,0].legend()
#         axes[0,0].grid(True, alpha=0.3)
#     else:
#         axes[0,0].text(0.5, 0.5, 'Training History\nNot Available', 
#                       ha='center', va='center', transform=axes[0,0].transAxes, 
#                       fontsize=12, style='italic')
#         axes[0,0].set_title('Training Loss History')
#         axes[0,0].set_facecolor('#f8f9fa')
    
#     # Plot 2: Sample Time Series Comparison
#     n_samples_show = min(6, real_data.shape[0], synthetic_data.shape[0])
#     time_steps = range(real_data.shape[1])
    
#     colors = plt.cm.Set1(np.linspace(0, 1, n_samples_show))
    
#     for i in range(n_samples_show):
#         axes[0,1].plot(time_steps, real_data[i, :, 0], 
#                       color=colors[i], alpha=0.7, linewidth=1.5, 
#                       label=f'Real {i+1}' if i < 3 else "")
#         axes[0,1].plot(time_steps, synthetic_data[i, :, 0], 
#                       color=colors[i], alpha=0.7, linewidth=1.5, linestyle='--',
#                       label=f'Synth {i+1}' if i < 3 else "")
    
#     axes[0,1].set_title(f'Time Series Comparison - {feature_names[0]}')
#     axes[0,1].set_xlabel('Time Steps')
#     axes[0,1].set_ylabel('Price ($)')
#     axes[0,1].legend()
#     axes[0,1].grid(True, alpha=0.3)
    
#     # Plot 3: Distribution Comparison
#     real_flat = real_data.reshape(-1, real_data.shape[-1])
#     synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
#     # Use reasonable bins based on data range
#     data_range = np.percentile(real_flat[:, 0], [1, 99])
#     bins = np.linspace(data_range[0], data_range[1], 50)
    
#     axes[0,2].hist(real_flat[:, 0], bins=bins, alpha=0.7, label='Real', 
#                   density=True, color='blue', edgecolor='black', linewidth=0.5)
#     axes[0,2].hist(synth_flat[:, 0], bins=bins, alpha=0.7, label='Synthetic', 
#                   density=True, color='red', edgecolor='black', linewidth=0.5)
#     axes[0,2].set_title(f'Distribution - {feature_names[0]}')
#     axes[0,2].set_xlabel('Price ($)')
#     axes[0,2].set_ylabel('Density')
#     axes[0,2].legend()
#     axes[0,2].grid(True, alpha=0.3)
    
#     # Plot 4-6: Correlation Matrices
#     n_features_show = min(5, len(feature_names))
    
#     # Real data correlations
#     real_corr = np.corrcoef(real_flat[:, :n_features_show].T)
#     im1 = axes[1,0].imshow(real_corr, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
#     axes[1,0].set_title('Real Data Correlations')
#     axes[1,0].set_xticks(range(n_features_show))
#     axes[1,0].set_xticklabels(feature_names[:n_features_show], rotation=45, ha='right')
#     axes[1,0].set_yticks(range(n_features_show))
#     axes[1,0].set_yticklabels(feature_names[:n_features_show])
#     plt.colorbar(im1, ax=axes[1,0], fraction=0.046, pad=0.04)
    
#     # Synthetic data correlations
#     synth_corr = np.corrcoef(synth_flat[:, :n_features_show].T)
#     im2 = axes[1,1].imshow(synth_corr, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
#     axes[1,1].set_title('Synthetic Data Correlations')
#     axes[1,1].set_xticks(range(n_features_show))
#     axes[1,1].set_xticklabels(feature_names[:n_features_show], rotation=45, ha='right')
#     axes[1,1].set_yticks(range(n_features_show))
#     axes[1,1].set_yticklabels(feature_names[:n_features_show])
#     plt.colorbar(im2, ax=axes[1,1], fraction=0.046, pad=0.04)
    
#     # Correlation differences
#     corr_diff = np.abs(real_corr - synth_corr)
#     im3 = axes[1,2].imshow(corr_diff, cmap='viridis', vmin=0, vmax=0.5, aspect='auto')
#     axes[1,2].set_title('Correlation Differences')
#     axes[1,2].set_xticks(range(n_features_show))
#     axes[1,2].set_xticklabels(feature_names[:n_features_show], rotation=45, ha='right')
#     axes[1,2].set_yticks(range(n_features_show))
#     axes[1,2].set_yticklabels(feature_names[:n_features_show])
#     plt.colorbar(im3, ax=axes[1,2], fraction=0.046, pad=0.04)
    
#     # Plot 7: Statistical Comparison
#     feature_idx = 0  # Compare first feature (Open price)
#     real_stats = [
#         np.mean(real_flat[:, feature_idx]),
#         np.std(real_flat[:, feature_idx]),
#         np.median(real_flat[:, feature_idx]),
#         np.percentile(real_flat[:, feature_idx], 25),
#         np.percentile(real_flat[:, feature_idx], 75)
#     ]
    
#     synth_stats = [
#         np.mean(synth_flat[:, feature_idx]),
#         np.std(synth_flat[:, feature_idx]),
#         np.median(synth_flat[:, feature_idx]),
#         np.percentile(synth_flat[:, feature_idx], 25),
#         np.percentile(synth_flat[:, feature_idx], 75)
#     ]
    
#     stats_names = ['Mean', 'Std', 'Median', 'Q1', 'Q3']
#     x_pos = np.arange(len(stats_names))
#     width = 0.35
    
#     axes[2,0].bar(x_pos - width/2, real_stats, width, label='Real', 
#                  color='blue', alpha=0.7, edgecolor='black')
#     axes[2,0].bar(x_pos + width/2, synth_stats, width, label='Synthetic', 
#                  color='red', alpha=0.7, edgecolor='black')
#     axes[2,0].set_title(f'Statistical Comparison - {feature_names[feature_idx]}')
#     axes[2,0].set_xlabel('Statistics')
#     axes[2,0].set_ylabel('Value ($)')
#     axes[2,0].set_xticks(x_pos)
#     axes[2,0].set_xticklabels(stats_names)
#     axes[2,0].legend()
#     axes[2,0].grid(True, alpha=0.3)
    
#     # Plot 8: Feature Relationship (if multiple features)
#     if len(feature_names) >= 2:
#         # Sample a subset for clarity
#         n_points = min(1000, real_flat.shape[0])
#         indices = np.random.choice(real_flat.shape[0], n_points, replace=False)
        
#         axes[2,1].scatter(real_flat[indices, 0], real_flat[indices, 1], 
#                          alpha=0.6, s=20, color='blue', label='Real')
#         axes[2,1].scatter(synth_flat[indices, 0], synth_flat[indices, 1], 
#                          alpha=0.6, s=20, color='red', label='Synthetic')
#         axes[2,1].set_xlabel(feature_names[0])
#         axes[2,1].set_ylabel(feature_names[1])
#         axes[2,1].set_title('Feature Relationship')
#         axes[2,1].legend()
#         axes[2,1].grid(True, alpha=0.3)
#     else:
#         axes[2,1].text(0.5, 0.5, 'Feature Relationship\nNot Available\n(Need ≥2 features)', 
#                       ha='center', va='center', transform=axes[2,1].transAxes, 
#                       fontsize=12, style='italic')
#         axes[2,1].set_title('Feature Relationship')
#         axes[2,1].set_facecolor('#f8f9fa')
    
#     # Plot 9: Data Quality Metrics
#     metrics = calculate_quality_metrics(real_flat, synth_flat)
    
#     metric_names = list(metrics.keys())
#     metric_values = list(metrics.values())
    
#     bars = axes[2,2].bar(metric_names, metric_values, 
#                         color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'],
#                         alpha=0.7, edgecolor='black')
    
#     axes[2,2].set_title('Data Quality Metrics')
#     axes[2,2].set_ylabel('Score (lower is better)')
#     axes[2,2].tick_params(axis='x', rotation=45)
#     axes[2,2].grid(True, alpha=0.3)
    
#     # Add value labels on bars
#     for bar, value in zip(bars, metric_values):
#         height = bar.get_height()
#         axes[2,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
#     plt.tight_layout()
    
#     # Save the plot
#     os.makedirs('outputs', exist_ok=True)
#     plt.savefig('outputs/comprehensive_timegan_analysis.png', dpi=300, bbox_inches='tight')
#     plt.savefig('outputs/comprehensive_timegan_analysis.pdf', bbox_inches='tight')
#     plt.show()
    
#     print("💾 Plot saved as: outputs/comprehensive_timegan_analysis.png")
#     print("✅ Plot generation completed successfully!")
    
#     # Print quality assessment
#     print("\n📈 QUALITY ASSESSMENT:")
#     if metrics['Mean_Diff'] < 100 and metrics['Correlation_Diff'] < 0.2:
#         print("🎉 EXCELLENT: High-quality synthetic data!")
#     elif metrics['Mean_Diff'] < 500 and metrics['Correlation_Diff'] < 0.3:
#         print("✅ GOOD: Acceptable synthetic data quality")
#     else:
#         print("⚠️ FAIR: Synthetic data generated but may need improvement")

# def calculate_quality_metrics(real_flat, synth_flat):
#     """Calculate data quality metrics"""
#     # Use only first few features for efficiency
#     n_features_use = min(5, real_flat.shape[1])
#     real_sub = real_flat[:, :n_features_use]
#     synth_sub = synth_flat[:, :n_features_use]
    
#     # 1. Mean difference
#     mean_diff = np.mean(np.abs(np.mean(real_sub, axis=0) - np.mean(synth_sub, axis=0)))
    
#     # 2. Standard deviation difference
#     std_diff = np.mean(np.abs(np.std(real_sub, axis=0) - np.std(synth_sub, axis=0)))
    
#     # 3. Correlation difference
#     real_corr = np.corrcoef(real_sub.T)
#     synth_corr = np.corrcoef(synth_sub.T)
#     corr_diff = np.mean(np.abs(real_corr - synth_corr))
    
#     # 4. Simple distribution distance (Wasserstein-like)
#     from scipy.stats import wasserstein_distance
#     wasserstein_dists = []
#     for i in range(n_features_use):
#         wasserstein_dists.append(wasserstein_distance(real_sub[:, i], synth_sub[:, i]))
#     wasserstein_avg = np.mean(wasserstein_dists)
    
#     return {
#         'Mean_Diff': mean_diff,
#         'Std_Diff': std_diff,
#         'Correlation_Diff': corr_diff,
#         'Wasserstein_Dist': wasserstein_avg
#     }

# if __name__ == "__main__":
#     generate_plots_from_existing_data()
    
    
    
    
    
    
    
    
    
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import os

# def enhanced_quick_plot():
#     """Enhanced quick plot with proper scaling and analysis"""
    
#     print("📊 Generating enhanced quick analysis...")
    
#     try:
#         # Load data
#         synthetic_data_scaled = np.load('outputs/synthetic/synthetic_bitcoin.npy')
        
#         # Load scaler and test data
#         with open('outputs/processed/scaler.pkl', 'rb') as f:
#             scaler = pickle.load(f)
#         test_data_scaled = np.load('outputs/processed/test.npy')
        
#         # Load feature names
#         with open('outputs/processed/features.txt', 'r') as f:
#             feature_names = [line.strip() for line in f]
        
#         print(f"✅ Loaded data shapes:")
#         print(f"   Test: {test_data_scaled.shape}")
#         print(f"   Synthetic: {synthetic_data_scaled.shape}")
#         print(f"   Features: {feature_names}")
        
#         # Apply inverse transformation
#         n_test, T_test, D_test = test_data_scaled.shape
#         n_synth, T_synth, D_synth = synthetic_data_scaled.shape
        
#         test_data = scaler.inverse_transform(
#             test_data_scaled.reshape(-1, D_test)
#         ).reshape(n_test, T_test, D_test)
        
#         synthetic_data = scaler.inverse_transform(
#             synthetic_data_scaled.reshape(-1, D_synth)
#         ).reshape(n_synth, T_synth, D_synth)
        
#         print(f"📊 After inverse scaling:")
#         print(f"   Real data range: ${np.min(test_data[:,:,0]):.2f} - ${np.max(test_data[:,:,0]):.2f}")
#         print(f"   Synthetic data range: ${np.min(synthetic_data[:,:,0]):.2f} - ${np.max(synthetic_data[:,:,0]):.2f}")
        
#         # Create comprehensive quick plot
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#         fig.suptitle('Bitcoin TimeGAN - Quick Analysis', fontsize=16, fontweight='bold')
        
#         # Plot 1: Sample comparison
#         n_samples = min(5, test_data.shape[0])
#         time_steps = range(T_test)
        
#         for i in range(n_samples):
#             axes[0,0].plot(time_steps, test_data[i, :, 0], 
#                           alpha=0.7, linewidth=2, label=f'Real {i+1}' if i == 0 else "")
#             axes[0,0].plot(time_steps, synthetic_data[i, :, 0], 
#                           alpha=0.7, linewidth=2, linestyle='--', 
#                           label=f'Synth {i+1}' if i == 0 else "")
        
#         axes[0,0].set_title(f'Sample Time Series - {feature_names[0]}')
#         axes[0,0].set_xlabel('Time Steps')
#         axes[0,0].set_ylabel('Price ($)')
#         axes[0,0].legend()
#         axes[0,0].grid(True, alpha=0.3)
        
#         # Plot 2: Distribution comparison
#         test_flat = test_data.reshape(-1, D_test)
#         synth_flat = synthetic_data.reshape(-1, D_synth)
        
#         # Use reasonable price range for bins
#         price_range = np.percentile(test_flat[:, 0], [0.5, 99.5])
#         bins = np.linspace(price_range[0], price_range[1], 40)
        
#         axes[0,1].hist(test_flat[:, 0], bins=bins, alpha=0.7, label='Real', 
#                       density=True, color='blue', edgecolor='black')
#         axes[0,1].hist(synth_flat[:, 0], bins=bins, alpha=0.7, label='Synthetic', 
#                       density=True, color='red', edgecolor='black')
#         axes[0,1].set_title('Price Distribution Comparison')
#         axes[0,1].set_xlabel('Price ($)')
#         axes[0,1].set_ylabel('Density')
#         axes[0,1].legend()
#         axes[0,1].grid(True, alpha=0.3)
        
#         # Plot 3: Statistical summary
#         stats_real = [
#             np.mean(test_flat[:, 0]),
#             np.std(test_flat[:, 0]),
#             np.median(test_flat[:, 0])
#         ]
        
#         stats_synth = [
#             np.mean(synth_flat[:, 0]),
#             np.std(synth_flat[:, 0]),
#             np.median(synth_flat[:, 0])
#         ]
        
#         stats_names = ['Mean', 'Std Dev', 'Median']
#         x_pos = np.arange(len(stats_names))
        
#         axes[1,0].bar(x_pos - 0.2, stats_real, 0.4, label='Real', 
#                      color='blue', alpha=0.7, edgecolor='black')
#         axes[1,0].bar(x_pos + 0.2, stats_synth, 0.4, label='Synthetic', 
#                      color='red', alpha=0.7, edgecolor='black')
#         axes[1,0].set_title('Statistical Comparison')
#         axes[1,0].set_xlabel('Statistics')
#         axes[1,0].set_ylabel('Price ($)')
#         axes[1,0].set_xticks(x_pos)
#         axes[1,0].set_xticklabels(stats_names)
#         axes[1,0].legend()
#         axes[1,0].grid(True, alpha=0.3)
        
#         # Plot 4: Quality metrics
#         mean_diff = abs(stats_real[0] - stats_synth[0])
#         std_diff = abs(stats_real[1] - stats_synth[1])
        
#         metrics = [mean_diff, std_diff]
#         metric_names = ['Mean Diff', 'Std Diff']
        
#         bars = axes[1,1].bar(metric_names, metrics, 
#                             color=['#ff9999', '#66b3ff'], 
#                             alpha=0.7, edgecolor='black')
#         axes[1,1].set_title('Quality Metrics (Lower is Better)')
#         axes[1,1].set_ylabel('Difference ($)')
#         axes[1,1].grid(True, alpha=0.3)
        
#         # Add value labels
#         for bar, value in zip(bars, metrics):
#             height = bar.get_height()
#             axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                           f'${value:.2f}', ha='center', va='bottom')
        
#         plt.tight_layout()
        
#         # Save plot
#         os.makedirs('outputs', exist_ok=True)
#         plt.savefig('outputs/enhanced_quick_analysis.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         print("\n📈 QUICK QUALITY ASSESSMENT:")
#         print(f"   Mean Price - Real: ${stats_real[0]:.2f}, Synthetic: ${stats_synth[0]:.2f}")
#         print(f"   Mean Difference: ${mean_diff:.2f}")
#         print(f"   Std Difference: ${std_diff:.2f}")
        
#         if mean_diff < 100 and std_diff < 50:
#             print("🎉 GOOD: Synthetic data captures real distribution well!")
#         else:
#             print("⚠️ MODERATE: Some differences detected in synthetic data")
            
#         print(f"💾 Plot saved as: outputs/enhanced_quick_analysis.png")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     enhanced_quick_plot()





import numpy as np
import pickle
import matplotlib.pyplot as plt

def diagnose_scaling_issue():
    """Diagnose why the synthetic data has extreme values"""
    
    print("🔍 DIAGNOSTIC: Checking data scaling issues...")
    
    try:
        # Load all data
        with open('outputs/processed/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        test_data = np.load('outputs/processed/test.npy')
        synthetic_data = np.load('outputs/synthetic/synthetic_bitcoin.npy')
        
        with open('outputs/processed/features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        print("\n📊 DATA SHAPES:")
        print(f"   Test data: {test_data.shape}")
        print(f"   Synthetic data: {synthetic_data.shape}")
        
        print("\n📈 DATA RANGES (Before Inverse Transform):")
        print(f"   Test data - Min: {np.min(test_data):.6f}, Max: {np.max(test_data):.6f}")
        print(f"   Synthetic data - Min: {np.min(synthetic_data):.6f}, Max: {np.max(synthetic_data):.6f}")
        
        print("\n📊 SCALER INFORMATION:")
        print(f"   Scaler type: {type(scaler)}")
        if hasattr(scaler, 'mean_'):
            print(f"   Scaler mean shape: {scaler.mean_.shape}")
            print(f"   Scaler mean values: {scaler.mean_}")
        if hasattr(scaler, 'scale_'):
            print(f"   Scaler scale shape: {scaler.scale_.shape}")
            print(f"   Scaler scale values: {scaler.scale_}")
        
        # Try inverse transform
        print("\n🔄 ATTEMPTING INVERSE TRANSFORM...")
        
        # Reshape for inverse transform
        n_test, T_test, D_test = test_data.shape
        n_synth, T_synth, D_synth = synthetic_data.shape
        
        test_flat = test_data.reshape(-1, D_test)
        synth_flat = synthetic_data.reshape(-1, D_synth)
        
        print(f"   Flat test shape: {test_flat.shape}")
        print(f"   Flat synth shape: {synth_flat.shape}")
        
        # Inverse transform
        test_inv = scaler.inverse_transform(test_flat)
        synth_inv = scaler.inverse_transform(synth_flat)
        
        print("\n📈 DATA RANGES (After Inverse Transform):")
        print(f"   Test data - Min: {np.min(test_inv):.2f}, Max: {np.max(test_inv):.2f}")
        print(f"   Synthetic data - Min: {np.min(synth_inv):.2f}, Max: {np.max(synth_inv):.2f}")
        
        # Check for extreme values
        test_extreme = np.percentile(test_inv, [0, 1, 50, 99, 100])
        synth_extreme = np.percentile(synth_inv, [0, 1, 50, 99, 100])
        
        print(f"\n📊 PERCENTILES - Test data:")
        print(f"   0%: {test_extreme[0]:.2f}, 1%: {test_extreme[1]:.2f}, 50%: {test_extreme[2]:.2f}, 99%: {test_extreme[3]:.2f}, 100%: {test_extreme[4]:.2f}")
        
        print(f"📊 PERCENTILES - Synthetic data:")
        print(f"   0%: {synth_extreme[0]:.2f}, 1%: {synth_extreme[1]:.2f}, 50%: {synth_extreme[2]:.2f}, 99%: {synth_extreme[3]:.2f}, 100%: {synth_extreme[4]:.2f}")
        
        # Create diagnostic plot
        create_diagnostic_plot(test_data, synthetic_data, test_inv, synth_inv, feature_names)
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_diagnostic_plot(test_data, synthetic_data, test_inv, synth_inv, feature_names):
    """Create diagnostic plot to understand the scaling issue"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Data Scaling Diagnostic Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Original scaled data distributions
    test_flat = test_data.reshape(-1, test_data.shape[-1])
    synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    axes[0,0].hist(test_flat[:, 0], bins=50, alpha=0.7, label='Real (scaled)', density=True, color='blue')
    axes[0,0].hist(synth_flat[:, 0], bins=50, alpha=0.7, label='Synthetic (scaled)', density=True, color='red')
    axes[0,0].set_title('Scaled Data Distribution')
    axes[0,0].set_xlabel('Scaled Values')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Inverse transformed data distributions
    axes[0,1].hist(test_inv[:, 0], bins=50, alpha=0.7, label='Real (inverse)', density=True, color='blue')
    axes[0,1].hist(synth_inv[:, 0], bins=50, alpha=0.7, label='Synthetic (inverse)', density=True, color='red')
    axes[0,1].set_title('Inverse Transformed Distribution')
    axes[0,1].set_xlabel('Original Scale Values')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Sample time series (scaled)
    n_samples = min(3, test_data.shape[0])
    for i in range(n_samples):
        axes[0,2].plot(test_data[i, :, 0], 'b-', alpha=0.7, label='Real Scaled' if i == 0 else "")
        axes[0,2].plot(synthetic_data[i, :, 0], 'r-', alpha=0.7, label='Synthetic Scaled' if i == 0 else "")
    axes[0,2].set_title('Scaled Time Series')
    axes[0,2].set_xlabel('Time Steps')
    axes[0,2].set_ylabel('Scaled Values')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Sample time series (inverse)
    test_inv_reshaped = test_inv.reshape(test_data.shape)
    synth_inv_reshaped = synth_inv.reshape(synthetic_data.shape)
    
    for i in range(n_samples):
        axes[1,0].plot(test_inv_reshaped[i, :, 0], 'b-', alpha=0.7, label='Real Inverse' if i == 0 else "")
        axes[1,0].plot(synth_inv_reshaped[i, :, 0], 'r-', alpha=0.7, label='Synthetic Inverse' if i == 0 else "")
    axes[1,0].set_title('Inverse Transformed Time Series')
    axes[1,0].set_xlabel('Time Steps')
    axes[1,0].set_ylabel('Original Values')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Statistical comparison
    stats_names = ['Mean', 'Std', 'Min', 'Max']
    test_stats = [np.mean(test_inv[:, 0]), np.std(test_inv[:, 0]), np.min(test_inv[:, 0]), np.max(test_inv[:, 0])]
    synth_stats = [np.mean(synth_inv[:, 0]), np.std(synth_inv[:, 0]), np.min(synth_inv[:, 0]), np.max(synth_inv[:, 0])]
    
    x_pos = np.arange(len(stats_names))
    width = 0.35
    
    axes[1,1].bar(x_pos - width/2, test_stats, width, label='Real', color='blue', alpha=0.7)
    axes[1,1].bar(x_pos + width/2, synth_stats, width, label='Synthetic', color='red', alpha=0.7)
    axes[1,1].set_title('Statistical Comparison (Inverse)')
    axes[1,1].set_xlabel('Statistics')
    axes[1,1].set_ylabel('Values')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(stats_names, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Data quality assessment
    mean_diff = abs(test_stats[0] - synth_stats[0]) / abs(test_stats[0]) * 100
    std_diff = abs(test_stats[1] - synth_stats[1]) / abs(test_stats[1]) * 100
    
    metrics = [mean_diff, std_diff]
    metric_names = ['Mean % Diff', 'Std % Diff']
    
    bars = axes[1,2].bar(metric_names, metrics, color=['orange', 'purple'], alpha=0.7)
    axes[1,2].set_title('Relative Differences (%)')
    axes[1,2].set_ylabel('Percentage Difference')
    axes[1,2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, metrics):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/scaling_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Diagnostic plot saved: outputs/scaling_diagnostic.png")

if __name__ == "__main__":
    diagnose_scaling_issue()