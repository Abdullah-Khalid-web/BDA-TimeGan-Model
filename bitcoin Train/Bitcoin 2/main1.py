# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import os
# from sklearn.preprocessing import MinMaxScaler
# import warnings
# warnings.filterwarnings('ignore')

# print("🚀 Bitcoin TimeGAN - FULL DATASET & 300 EPOCHS")
# print("=" * 70)

# # Create directories
# os.makedirs('results/plots', exist_ok=True)
# os.makedirs('models', exist_ok=True)

# # ===== ENHANCED CONFIGURATION =====
# SAMPLE_SIZE = 1000
# MAX_SEQUENCES = 10000  # Increased for full dataset
# EPOCHS = 30  # Increased to 300 epochs
# BATCH_SIZE = 64  # Increased batch size
# HIDDEN_DIM = 64  # Increased capacity
# LATENT_DIM = 16  # Increased latent space
# SEQ_LEN = 24
# LEARNING_RATE = 0.0005  # Lower for stable training
# # ==================================

# # 🔥 Step 1: Load Full Bitcoin Dataset
# print("\n📊 Step 1: Loading Full Bitcoin Dataset...")

# # def load_data(file_path):
# #     """Load entire Bitcoin dataset efficiently"""
# #     try:
# #         print(f"📁 Loading full dataset from {file_path}...")
        
# #         # Check file size
# #         file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
# #         print(f"📏 File size: {file_size:.2f} GB")
        
# #         # First inspect the structure
# #         with open(file_path, 'r') as f:
# #             first_line = f.readline().strip()
# #             print(f"📝 First line: {first_line}")
        
# #         # Try to read with proper column detection
# #         df_sample = pd.read_csv(file_path, nrows=5)
# #         print(f"📊 Detected columns: {list(df_sample.columns)}")
        
# #         # Load full dataset with optimized data types
# #         if len(df_sample.columns) >= 5:
# #             print("✅ Loading with detected headers...")
# #             df = pd.read_csv(file_path)
# #         else:
# #             print("⚠️ Loading without headers...")
# #             df = pd.read_csv(file_path, header=None)
# #             if df.shape[1] >= 5:
# #                 df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'][:df.shape[1]]
        
# #         print(f"✅ Full dataset loaded: {df.shape}")
# #         print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
# #         return df
        
# #     except Exception as e:
# #         print(f"❌ Error loading file: {e}")
# #         return None

# def load_data(file_path, sample_size=None, random_state=42):
#     """Load Bitcoin dataset efficiently with optional sampling for testing"""
#     try:
#         print(f"📁 Loading dataset from {file_path}...")
        
#         # Check file size
#         file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
#         print(f"📏 File size: {file_size:.2f} GB")
        
#         # First inspect the structure
#         with open(file_path, 'r') as f:
#             first_line = f.readline().strip()
#             print(f"📝 First line: {first_line}")
        
#         # Try to read with proper column detection
#         df_sample = pd.read_csv(file_path, nrows=5)
#         print(f"📊 Detected columns: {list(df_sample.columns)}")
        
#         # Load data with or without sampling
#         if sample_size:
#             print(f"🎯 Sampling {sample_size:,} rows for testing...")
            
#             # Get total rows to sample from (excluding header)
#             total_rows = sum(1 for line in open(file_path)) - 1
            
#             if sample_size >= total_rows:
#                 print("⚠️ Sample size larger than dataset, loading full data...")
#                 if len(df_sample.columns) >= 5:
#                     df = pd.read_csv(file_path)
#                 else:
#                     df = pd.read_csv(file_path, header=None)
#             else:
#                 # Sample random rows
#                 skip_rows = np.random.choice(total_rows, total_rows - sample_size, replace=False)
#                 skip_rows = [x + 1 for x in skip_rows]  # +1 to skip header
                
#                 if len(df_sample.columns) >= 5:
#                     df = pd.read_csv(file_path, skiprows=skip_rows)
#                 else:
#                     df = pd.read_csv(file_path, header=None, skiprows=skip_rows)
#         else:
#             print("✅ Loading full dataset...")
#             if len(df_sample.columns) >= 5:
#                 df = pd.read_csv(file_path)
#             else:
#                 df = pd.read_csv(file_path, header=None)
        
#         # Assign column names if needed
#         if len(df_sample.columns) < 5 or 'Open' not in df.columns:
#             print("⚠️ Assigning column names...")
#             if df.shape[1] == 6:
#                 df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
#             elif df.shape[1] == 5:
#                 df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#             else:
#                 df.columns = [f'col_{i}' for i in range(df.shape[1])]
        
#         print(f"✅ Dataset loaded: {df.shape}")
#         print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
#         # Show sample info
#         print(f"📈 Data sample (first 3 rows):")
#         print(df.head(3))
#         print(f"📊 Data types:")
#         print(df.dtypes)
        
#         return df
        
#     except Exception as e:
#         print(f"❌ Error loading file: {e}")
#         return None

# # Load full data
# df = load_data('../data/bitcoin.csv')
# if df is None:
#     print("❌ Could not load bitcoin.csv")
#     exit()

# # 🧹 Step 2: Enhanced Data Preprocessing
# print("\n🧹 Step 2: Preprocessing Full Dataset...")

# def preprocess_full_data(df):
#     """Enhanced preprocessing for full dataset"""
    
#     print("🔍 Analyzing full dataset structure...")
    
#     # Identify numeric columns
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     print(f"📊 Numeric columns found: {numeric_cols}")
    
#     # Use Bitcoin columns if available
#     bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
#     available_bitcoin_cols = [col for col in bitcoin_cols if col in df.columns]
    
#     if available_bitcoin_cols:
#         print(f"✅ Using Bitcoin columns: {available_bitcoin_cols}")
#         features_to_use = available_bitcoin_cols
#     else:
#         if 'Timestamp' in numeric_cols:
#             numeric_cols.remove('Timestamp')
#         features_to_use = numeric_cols[:5]
#         print(f"⚠️ Using first 5 numeric columns: {features_to_use}")
    
#     if not features_to_use:
#         print("❌ No numeric features found!")
#         return None, None, None
    
#     # Extract and clean data
#     data = df[features_to_use].values
#     data = np.nan_to_num(data)
#     data = np.clip(data, -1e10, 1e10)
    
#     print(f"📊 Full data stats:")
#     print(f"   Shape: {data.shape}")
#     print(f"   Min: {np.min(data, axis=0)}")
#     print(f"   Max: {np.max(data, axis=0)}")
#     print(f"   Mean: {np.mean(data, axis=0)}")
    
#     # Normalize
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data)
    
#     print(f"✅ Scaled data shape: {data_scaled.shape}")
    
#     return data_scaled, scaler, features_to_use

# data_scaled, scaler, feature_names = preprocess_full_data(df)

# # ⏱️ Step 3: Create Enhanced Sequences
# print(f"\n⏱️ Step 3: Creating {MAX_SEQUENCES} sequences from full dataset...")

# def create_sequences_enhanced(data, seq_len=SEQ_LEN, max_seq=MAX_SEQUENCES):
#     """Create sequences with better sampling"""
#     n_possible = len(data) - seq_len
#     n_sequences = min(max_seq, n_possible)
    
#     # Use stratified sampling to get diverse sequences
#     step_size = max(1, n_possible // n_sequences)
#     indices = np.arange(0, n_possible, step_size)[:n_sequences]
    
#     sequences = np.array([data[i:i+seq_len] for i in indices])
    
#     print(f"✅ Created {len(sequences)} sequences of shape {sequences.shape}")
#     print(f"💾 Memory usage: {sequences.nbytes / 1024 / 1024:.1f} MB")
    
#     return sequences

# sequences = create_sequences_enhanced(data_scaled)

# # 🧠 Step 4: Build Enhanced TimeGAN Model
# print("\n🧠 Step 4: Building Enhanced TimeGAN Model...")

# class EnhancedTimeGAN:
#     def __init__(self, seq_len, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
#         self.seq_len = seq_len
#         self.n_features = n_features
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
        
#         self.generator = self._build_enhanced_generator()
#         self.discriminator = self._build_enhanced_discriminator()
        
#         self.g_optimizer = keras.optimizers.Adam(LEARNING_RATE)
#         self.d_optimizer = keras.optimizers.Adam(LEARNING_RATE * 0.5)
        
#         print(f"✅ Enhanced TimeGAN built!")
#         print(f"   Generator: {self.generator.count_params():,} params")
#         print(f"   Discriminator: {self.discriminator.count_params():,} params")
    
#     def _build_enhanced_generator(self):
#         return keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True, 
#                        dropout=0.2, recurrent_dropout=0.2,
#                        input_shape=(self.seq_len, self.latent_dim)),
#             layers.BatchNormalization(),
#             layers.LSTM(self.hidden_dim, return_sequences=True, 
#                        dropout=0.2, recurrent_dropout=0.2),
#             layers.BatchNormalization(),
#             layers.Dense(self.hidden_dim // 2, activation='relu'),
#             layers.Dense(self.n_features, activation='sigmoid')
#         ], name="Enhanced_Generator")
    
#     def _build_enhanced_discriminator(self):
#         return keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True,
#                        dropout=0.2, recurrent_dropout=0.2,
#                        input_shape=(self.seq_len, self.n_features)),
#             layers.BatchNormalization(),
#             layers.LSTM(self.hidden_dim // 2, return_sequences=True,
#                        dropout=0.2, recurrent_dropout=0.2),
#             layers.BatchNormalization(),
#             layers.LSTM(self.hidden_dim // 4, return_sequences=False,
#                        dropout=0.2, recurrent_dropout=0.2),
#             layers.Dense(1, activation='sigmoid')
#         ], name="Enhanced_Discriminator")

# timegan = EnhancedTimeGAN(SEQ_LEN, sequences.shape[2])

# # 🏋️ Step 5: Enhanced Training with 300 Epochs
# print(f"\n🏋️ Step 5: Training for {EPOCHS} epochs...")

# # Callbacks for better training
# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='loss', patience=20, restore_best_weights=True
# )

# reduce_lr = keras.callbacks.ReduceLROnPlateau(
#     monitor='loss', factor=0.5, patience=10, min_lr=0.0001
# )

# @tf.function
# def enhanced_train_step(generator, discriminator, g_opt, d_opt, real_batch, seq_len, latent_dim):
#     batch_size = tf.shape(real_batch)[0]
    
#     # Generate noise with different variances for diversity
#     noise = tf.random.normal((batch_size, seq_len, latent_dim), stddev=0.5)
    
#     # Train Discriminator
#     with tf.GradientTape() as d_tape:
#         fake_data = generator(noise, training=True)
#         real_output = discriminator(real_batch, training=True)
#         fake_output = discriminator(fake_data, training=True)
        
#         d_loss_real = tf.reduce_mean(keras.losses.binary_crossentropy(
#             tf.ones_like(real_output), real_output))
#         d_loss_fake = tf.reduce_mean(keras.losses.binary_crossentropy(
#             tf.zeros_like(fake_output), fake_output))
#         d_loss = (d_loss_real + d_loss_fake) * 0.5
        
#         # Add gradient penalty for stability
#         epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
#         interpolated = epsilon * real_batch + (1 - epsilon) * fake_data
#         with tf.GradientTape() as gp_tape:
#             gp_tape.watch(interpolated)
#             interpolated_output = discriminator(interpolated, training=True)
        
#         gradients = gp_tape.gradient(interpolated_output, interpolated)
#         gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
#         gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
#         d_loss += 10.0 * gradient_penalty
    
#     d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
#     d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
#     # Train Generator
#     with tf.GradientTape() as g_tape:
#         fake_data = generator(noise, training=True)
#         fake_output = discriminator(fake_data, training=True)
#         g_loss = tf.reduce_mean(keras.losses.binary_crossentropy(
#             tf.ones_like(fake_output), fake_output))
        
#         # Add feature matching loss
#         real_features = tf.reduce_mean(real_output, axis=0)
#         fake_features = tf.reduce_mean(fake_output, axis=0)
#         feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
        
#         g_loss += feature_loss
    
#     g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
#     g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    
#     return d_loss, g_loss

# def enhanced_train_model(timegan, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE):
#     d_losses, g_losses = [], []
#     n_batches = max(1, len(sequences) // batch_size)
    
#     print("🚀 Enhanced training started...")
#     print(f"📊 Total batches per epoch: {n_batches}")
#     print(f"📈 Monitoring training progress...")
    
#     for epoch in range(epochs):
#         epoch_d_loss, epoch_g_loss = 0, 0
        
#         # Shuffle data each epoch
#         indices = np.random.permutation(len(sequences))
        
#         for batch in range(n_batches):
#             start_idx = batch * batch_size
#             end_idx = start_idx + batch_size
#             batch_indices = indices[start_idx:end_idx]
#             real_batch = sequences[batch_indices]
            
#             d_loss, g_loss = enhanced_train_step(
#                 timegan.generator, timegan.discriminator,
#                 timegan.g_optimizer, timegan.d_optimizer,
#                 real_batch, SEQ_LEN, timegan.latent_dim
#             )
            
#             epoch_d_loss += d_loss.numpy()
#             epoch_g_loss += g_loss.numpy()
        
#         avg_d_loss = epoch_d_loss / n_batches
#         avg_g_loss = epoch_g_loss / n_batches
#         d_losses.append(avg_d_loss)
#         g_losses.append(avg_g_loss)
        
#         # Enhanced progress reporting
#         if epoch % 20 == 0 or epoch == epochs - 1:
#             progress = (epoch + 1) / epochs * 100
#             print(f"📊 Epoch {epoch+1:3d}/{epochs} [{progress:5.1f}%] | "
#                   f"D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
            
#             # Early stopping check
#             if len(d_losses) > 30 and np.mean(d_losses[-10:]) < 0.1:
#                 print("🎯 Early stopping: Model converged!")
#                 break
    
#     return d_losses, g_losses

# d_losses, g_losses = enhanced_train_model(timegan, sequences)
# print("✅ Enhanced training completed!")

# # 💾 Save Enhanced Model
# print("\n💾 Saving Enhanced Model...")
# timegan.generator.save('models/timegan_generator_enhanced.h5')
# timegan.discriminator.save('models/timegan_discriminator_enhanced.h5')
# print("✅ Both generator and discriminator saved!")

# # 🎨 Generate High-Quality Synthetic Data
# print("\n🎨 Generating High-Quality Synthetic Data...")

# def generate_enhanced_synthetic(generator, n_samples, seq_len, latent_dim):
#     """Generate synthetic data with temperature sampling"""
#     # Generate with different noise levels for diversity
#     temperatures = [0.7, 1.0, 1.3]
#     synthetic_data = []
    
#     for temp in temperatures:
#         noise = np.random.normal(0, temp, (n_samples // len(temperatures), seq_len, latent_dim))
#         generated = generator.predict(noise, verbose=0)
#         synthetic_data.append(generated)
    
#     return np.vstack(synthetic_data)

# n_synthetic = min(5000, len(sequences))
# synthetic_data = generate_enhanced_synthetic(timegan.generator, n_synthetic, SEQ_LEN, timegan.latent_dim)
# print(f"✅ Generated {len(synthetic_data)} high-quality synthetic sequences")

# # Denormalize data
# synthetic_denorm = scaler.inverse_transform(
#     synthetic_data.reshape(-1, synthetic_data.shape[-1])
# ).reshape(synthetic_data.shape)

# real_sample = sequences[:n_synthetic]
# real_denorm = scaler.inverse_transform(
#     real_sample.reshape(-1, real_sample.shape[-1])
# ).reshape(real_sample.shape)

# # 📊 Step 6: Enhanced Visualizations
# print("\n📊 Creating Enhanced Visualizations...")

# def create_enhanced_visualizations(real_data, synthetic_data, d_losses, g_losses, feature_names):
#     plt.style.use('seaborn-v0_8')
#     fig = plt.figure(figsize=(25, 20))
    
#     # 1. Training Loss with Smoothing
#     plt.subplot(3, 3, 1)
#     # Apply smoothing
#     window = max(1, len(d_losses) // 50)
#     d_smooth = pd.Series(d_losses).rolling(window=window, center=True).mean()
#     g_smooth = pd.Series(g_losses).rolling(window=window, center=True).mean()
    
#     plt.plot(d_losses, 'r-', alpha=0.3, label='Discriminator Raw')
#     plt.plot(g_losses, 'b-', alpha=0.3, label='Generator Raw')
#     plt.plot(d_smooth, 'r-', linewidth=3, label='Discriminator Smooth')
#     plt.plot(g_smooth, 'b-', linewidth=3, label='Generator Smooth')
#     plt.title('Enhanced Training Loss', fontsize=14, fontweight='bold')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.yscale('log')
    
#     # 2. Multiple Sample Comparisons
#     plt.subplot(3, 3, 2)
#     n_samples_show = min(5, len(real_data))
#     for i in range(n_samples_show):
#         plt.plot(real_data[i, :, 0], 'g-', alpha=0.7, linewidth=1, 
#                 label='Real' if i == 0 else "")
#         plt.plot(synthetic_data[i, :, 0], 'orange', alpha=0.7, linewidth=1, 
#                 label='Synthetic' if i == 0 else "")
#     plt.title(f'Multiple {feature_names[0]} Samples', fontsize=14, fontweight='bold')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 3. Enhanced Distribution Comparison
#     plt.subplot(3, 3, 3)
#     real_flat = real_data.reshape(-1, real_data.shape[-1])
#     synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
#     for i, feature in enumerate(feature_names[:3]):  # First 3 features
#         plt.hist(real_flat[:, i], bins=50, alpha=0.5, label=f'Real {feature}', density=True)
#         plt.hist(synth_flat[:, i], bins=50, alpha=0.5, label=f'Synthetic {feature}', density=True)
    
#     plt.title('Multi-Feature Distributions', fontsize=14, fontweight='bold')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 4. Feature Correlation Heatmaps
#     plt.subplot(3, 3, 4)
#     real_corr = np.corrcoef(real_flat[:, :5].T)
#     plt.imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.colorbar(label='Correlation')
#     plt.title('Real Data Correlations', fontsize=14, fontweight='bold')
#     plt.xticks(range(min(5, len(feature_names))), feature_names[:5], rotation=45)
#     plt.yticks(range(min(5, len(feature_names))), feature_names[:5])
    
#     plt.subplot(3, 3, 5)
#     synth_corr = np.corrcoef(synth_flat[:, :5].T)
#     plt.imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.colorbar(label='Correlation')
#     plt.title('Synthetic Data Correlations', fontsize=14, fontweight='bold')
#     plt.xticks(range(min(5, len(feature_names))), feature_names[:5], rotation=45)
#     plt.yticks(range(min(5, len(feature_names))), feature_names[:5])
    
#     # 5. Correlation Difference
#     plt.subplot(3, 3, 6)
#     corr_diff = np.abs(real_corr - synth_corr)
#     plt.imshow(corr_diff, cmap='hot', vmin=0, vmax=0.5)
#     plt.colorbar(label='Absolute Difference')
#     plt.title('Correlation Differences', fontsize=14, fontweight='bold')
#     plt.xticks(range(min(5, len(feature_names))), feature_names[:5], rotation=45)
#     plt.yticks(range(min(5, len(feature_names))), feature_names[:5])
    
#     # 6. Temporal Patterns
#     plt.subplot(3, 3, 7)
#     real_mean = np.mean(real_data[:, :, 0], axis=0)
#     real_std = np.std(real_data[:, :, 0], axis=0)
#     synth_mean = np.mean(synthetic_data[:, :, 0], axis=0)
#     synth_std = np.std(synthetic_data[:, :, 0], axis=0)
    
#     plt.plot(real_mean, 'g-', label='Real Mean', linewidth=2)
#     plt.fill_between(range(len(real_mean)), real_mean - real_std, real_mean + real_std, 
#                     alpha=0.3, color='green', label='Real Std')
#     plt.plot(synth_mean, 'orange', label='Synthetic Mean', linewidth=2)
#     plt.fill_between(range(len(synth_mean)), synth_mean - synth_std, synth_mean + synth_std, 
#                     alpha=0.3, color='orange', label='Synthetic Std')
#     plt.title('Temporal Pattern Comparison', fontsize=14, fontweight='bold')
#     plt.xlabel('Time Steps')
#     plt.ylabel(feature_names[0])
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 7. Feature Relationships Scatter
#     plt.subplot(3, 3, 8)
#     if len(feature_names) >= 2:
#         plt.scatter(real_flat[:1000, 0], real_flat[:1000, 1], alpha=0.5, 
#                    label='Real', s=10, color='green')
#         plt.scatter(synth_flat[:1000, 0], synth_flat[:1000, 1], alpha=0.5, 
#                    label='Synthetic', s=10, color='orange')
#         plt.xlabel(feature_names[0])
#         plt.ylabel(feature_names[1])
#         plt.title('Feature Relationship', fontsize=14, fontweight='bold')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
    
#     # 8. Quality Metrics
#     plt.subplot(3, 3, 9)
#     metrics = {
#         'Mean Diff': np.mean(np.abs(np.mean(real_flat, axis=0) - np.mean(synth_flat, axis=0))),
#         'Std Diff': np.mean(np.abs(np.std(real_flat, axis=0) - np.std(synth_flat, axis=0))),
#         'Corr Diff': np.mean(np.abs(real_corr - synth_corr)),
#         'Final D Loss': d_losses[-1],
#         'Final G Loss': g_losses[-1]
#     }
    
#     plt.bar(range(len(metrics)), list(metrics.values()), color=['blue', 'orange', 'green', 'red', 'purple'])
#     plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
#     plt.title('Quality Metrics', fontsize=14, fontweight='bold')
#     plt.ylabel('Value')
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('results/plots/enhanced_bitcoin_timegan.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return metrics

# metrics = create_enhanced_visualizations(real_denorm, synthetic_denorm, d_losses, g_losses, feature_names)

# # 📈 Save Enhanced Synthetic Data
# print("\n📈 Saving Enhanced Synthetic Data...")
# synthetic_flat = synthetic_denorm.reshape(-1, synthetic_denorm.shape[-1])
# synthetic_df = pd.DataFrame(synthetic_flat, columns=feature_names)
# synthetic_df.to_csv('results/synthetic_bitcoin_enhanced.csv', index=False)
# print(f"✅ Enhanced synthetic data saved: {synthetic_df.shape}")

# # 💾 Save Training History
# training_history = pd.DataFrame({
#     'epoch': range(1, len(d_losses) + 1),
#     'discriminator_loss': d_losses,
#     'generator_loss': g_losses
# })
# training_history.to_csv('results/training_history_300_epochs.csv', index=False)
# print("✅ Training history saved!")

# # 🎯 Final Enhanced Report
# print("\n" + "="*60)
# print("🎉 ENHANCED TIMEGAN TRAINING COMPLETE!")
# print("="*60)
# print(f"📊 Dataset: {len(df):,} samples (FULL DATASET)")
# print(f"⚡ Training: {len(d_losses)}/{EPOCHS} epochs completed")
# print(f"📈 Sequence shape: {sequences.shape}")
# print(f"🔧 Model: Enhanced LSTM with {HIDDEN_DIM} hidden units")
# print(f"📉 Final D_loss: {d_losses[-1]:.4f}")
# print(f"📈 Final G_loss: {g_losses[-1]:.4f}")
# print(f"🎯 Quality Metrics:")
# print(f"   • Mean Difference: {metrics['Mean Diff']:.6f}")
# print(f"   • Std Difference: {metrics['Std Diff']:.6f}")
# print(f"   • Correlation Difference: {metrics['Corr Diff']:.6f}")
# print(f"💾 Outputs:")
# print(f"   • models/timegan_generator_enhanced.h5")
# print(f"   • results/synthetic_bitcoin_enhanced.csv")
# print(f"   • results/training_history_300_epochs.csv")
# print(f"   • results/plots/enhanced_bitcoin_timegan.png")
# print("\n✅ Success! High-quality synthetic Bitcoin data generated!")









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("🚀 Bitcoin TimeGAN - FULL DATASET & 300 EPOCHS")
print("=" * 70)

# Create directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== ENHANCED CONFIGURATION =====
MAX_SEQUENCES = 10000  # Increased for full dataset
EPOCHS = 300  # Increased to 300 epochs
# MAX_SEQUENCES = 10000  # Increased for full dataset
# EPOCHS = 300  # Increased to 300 epochs
BATCH_SIZE = 64  # Increased batch size
HIDDEN_DIM = 64  # Increased capacity
LATENT_DIM = 16  # Increased latent space
SEQ_LEN = 24
LEARNING_RATE = 0.0005  # Lower for stable training
# ==================================

# 🔥 Step 1: Load Full Bitcoin Dataset
print("\n📊 Step 1: Loading Full Bitcoin Dataset...")
# change the none to any 
def load_data(file_path, sample_size=None, random_state=42):
    """Load Bitcoin dataset efficiently with optional sampling for testing"""
    try:
        print(f"📁 Loading dataset from {file_path}...")
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
        print(f"📏 File size: {file_size:.2f} GB")
        
        # First inspect the structure
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            print(f"📝 First line: {first_line}")
        
        # Try to read with proper column detection
        df_sample = pd.read_csv(file_path, nrows=5)
        print(f"📊 Detected columns: {list(df_sample.columns)}")
        
        # Load data with or without sampling
        if sample_size:
            print(f"🎯 Sampling {sample_size:,} rows for testing...")
            
            # Get total rows to sample from (excluding header)
            total_rows = sum(1 for line in open(file_path)) - 1
            
            if sample_size >= total_rows:
                print("⚠️ Sample size larger than dataset, loading full data...")
                if len(df_sample.columns) >= 5:
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(file_path, header=None)
            else:
                # Sample random rows
                skip_rows = np.random.choice(total_rows, total_rows - sample_size, replace=False)
                skip_rows = [x + 1 for x in skip_rows]  # +1 to skip header
                
                if len(df_sample.columns) >= 5:
                    df = pd.read_csv(file_path, skiprows=skip_rows)
                else:
                    df = pd.read_csv(file_path, header=None, skiprows=skip_rows)
        else:
            print("✅ Loading full dataset...")
            if len(df_sample.columns) >= 5:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path, header=None)
        
        # Assign column names if needed
        if len(df_sample.columns) < 5 or 'Open' not in df.columns:
            print("⚠️ Assigning column names...")
            if df.shape[1] == 6:
                df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            elif df.shape[1] == 5:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                df.columns = [f'col_{i}' for i in range(df.shape[1])]
        
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Show sample info
        print(f"📈 Data sample (first 3 rows):")
        print(df.head(3))
        print(f"📊 Data types:")
        print(df.dtypes)
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

# Load full data
df = load_data('../data/bitcoin.csv')  # Remove sample_size for full dataset

# 🧹 Step 2: Enhanced Data Preprocessing with Data Type Fix
print("\n🧹 Step 2: Preprocessing Full Dataset...")

def preprocess_full_data(df):
    """Enhanced preprocessing for full dataset with proper data types"""
    
    print("🔍 Analyzing full dataset structure...")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 Numeric columns found: {numeric_cols}")
    
    # Use Bitcoin columns if available
    bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_bitcoin_cols = [col for col in bitcoin_cols if col in df.columns]
    
    if available_bitcoin_cols:
        print(f"✅ Using Bitcoin columns: {available_bitcoin_cols}")
        features_to_use = available_bitcoin_cols
    else:
        if 'Timestamp' in numeric_cols:
            numeric_cols.remove('Timestamp')
        features_to_use = numeric_cols[:5]
        print(f"⚠️ Using first 5 numeric columns: {features_to_use}")
    
    if not features_to_use:
        print("❌ No numeric features found!")
        return None, None, None
    
    # Extract and clean data
    data = df[features_to_use].values
    data = np.nan_to_num(data)
    data = np.clip(data, -1e10, 1e10)
    
    print(f"📊 Full data stats:")
    print(f"   Shape: {data.shape}")
    print(f"   Min: {np.min(data, axis=0)}")
    print(f"   Max: {np.max(data, axis=0)}")
    print(f"   Mean: {np.mean(data, axis=0)}")
    
    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Convert to float32 for TensorFlow compatibility
    data_scaled = data_scaled.astype(np.float32)
    
    print(f"✅ Scaled data shape: {data_scaled.shape}")
    print(f"🔧 Data type: {data_scaled.dtype}")
    
    return data_scaled, scaler, features_to_use

data_scaled, scaler, feature_names = preprocess_full_data(df)

# ⏱️ Step 3: Create Enhanced Sequences with Data Type Fix
print(f"\n⏱️ Step 3: Creating {MAX_SEQUENCES} sequences from full dataset...")

def create_sequences_enhanced(data, seq_len=SEQ_LEN, max_seq=MAX_SEQUENCES):
    """Create sequences with better sampling and proper data types"""
    n_possible = len(data) - seq_len
    n_sequences = min(max_seq, n_possible)
    
    # Use stratified sampling to get diverse sequences
    step_size = max(1, n_possible // n_sequences)
    indices = np.arange(0, n_possible, step_size)[:n_sequences]
    
    sequences = np.array([data[i:i+seq_len] for i in indices], dtype=np.float32)
    
    print(f"✅ Created {len(sequences)} sequences of shape {sequences.shape}")
    print(f"💾 Memory usage: {sequences.nbytes / 1024 / 1024:.1f} MB")
    print(f"🔧 Data type: {sequences.dtype}")
    
    return sequences

sequences = create_sequences_enhanced(data_scaled)

# 🧠 Step 4: Build Enhanced TimeGAN Model
print("\n🧠 Step 4: Building Enhanced TimeGAN Model...")

class EnhancedTimeGAN:
    def __init__(self, seq_len, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.generator = self._build_enhanced_generator()
        self.discriminator = self._build_enhanced_discriminator()
        
        self.g_optimizer = keras.optimizers.Adam(LEARNING_RATE)
        self.d_optimizer = keras.optimizers.Adam(LEARNING_RATE * 0.5)
        
        print(f"✅ Enhanced TimeGAN built!")
        print(f"   Generator: {self.generator.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
    
    def _build_enhanced_generator(self):
        return keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, 
                       dropout=0.2, recurrent_dropout=0.2,
                       input_shape=(self.seq_len, self.latent_dim)),
            layers.BatchNormalization(),
            layers.LSTM(self.hidden_dim, return_sequences=True, 
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            layers.Dense(self.hidden_dim // 2, activation='relu'),
            layers.Dense(self.n_features, activation='sigmoid')
        ], name="Enhanced_Generator")
    
    def _build_enhanced_discriminator(self):
        return keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2,
                       input_shape=(self.seq_len, self.n_features)),
            layers.BatchNormalization(),
            layers.LSTM(self.hidden_dim // 2, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(self.hidden_dim // 4, return_sequences=False,
                       dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(1, activation='sigmoid')
        ], name="Enhanced_Discriminator")

timegan = EnhancedTimeGAN(SEQ_LEN, sequences.shape[2])

# 🏋️ Step 5: Enhanced Training with 300 Epochs - FIXED VERSION
print(f"\n🏋️ Step 5: Training for {EPOCHS} epochs...")

@tf.function
def enhanced_train_step(generator, discriminator, g_opt, d_opt, real_batch, seq_len, latent_dim):
    batch_size = tf.shape(real_batch)[0]
    
    # Generate noise with proper data type
    noise = tf.random.normal((batch_size, seq_len, latent_dim), stddev=0.5, dtype=tf.float32)
    
    # Train Discriminator
    with tf.GradientTape() as d_tape:
        fake_data = generator(noise, training=True)
        real_output = discriminator(real_batch, training=True)
        fake_output = discriminator(fake_data, training=True)
        
        d_loss_real = tf.reduce_mean(keras.losses.binary_crossentropy(
            tf.ones_like(real_output), real_output))
        d_loss_fake = tf.reduce_mean(keras.losses.binary_crossentropy(
            tf.zeros_like(fake_output), fake_output))
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        # Add gradient penalty for stability - FIXED DATA TYPE
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
        interpolated = epsilon * real_batch + (1 - epsilon) * fake_data
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = discriminator(interpolated, training=True)
        
        gradients = gp_tape.gradient(interpolated_output, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        d_loss += 10.0 * gradient_penalty
    
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # Train Generator
    with tf.GradientTape() as g_tape:
        fake_data = generator(noise, training=True)
        fake_output = discriminator(fake_data, training=True)
        g_loss = tf.reduce_mean(keras.losses.binary_crossentropy(
            tf.ones_like(fake_output), fake_output))
        
        # Add feature matching loss
        real_features = tf.reduce_mean(real_output, axis=0)
        fake_features = tf.reduce_mean(fake_output, axis=0)
        feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
        
        g_loss += feature_loss
    
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss

def enhanced_train_model(timegan, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE):
    d_losses, g_losses = [], []
    n_batches = max(1, len(sequences) // batch_size)
    
    print("🚀 Enhanced training started...")
    print(f"📊 Total batches per epoch: {n_batches}")
    print(f"📈 Monitoring training progress...")
    
    # Ensure sequences are float32
    sequences = sequences.astype(np.float32)
    
    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss = 0, 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(len(sequences))
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            real_batch = sequences[batch_indices]
            
            # Ensure batch is float32
            real_batch = tf.convert_to_tensor(real_batch, dtype=tf.float32)
            
            d_loss, g_loss = enhanced_train_step(
                timegan.generator, timegan.discriminator,
                timegan.g_optimizer, timegan.d_optimizer,
                real_batch, SEQ_LEN, timegan.latent_dim
            )
            
            epoch_d_loss += d_loss.numpy()
            epoch_g_loss += g_loss.numpy()
        
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        # Enhanced progress reporting
        if epoch % 20 == 0 or epoch == epochs - 1:
            progress = (epoch + 1) / epochs * 100
            print(f"📊 Epoch {epoch+1:3d}/{epochs} [{progress:5.1f}%] | "
                  f"D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
            
            # Early stopping check
            if len(d_losses) > 30 and np.mean(d_losses[-10:]) < 0.1:
                print("🎯 Early stopping: Model converged!")
                break
    
    return d_losses, g_losses

# Start training
d_losses, g_losses = enhanced_train_model(timegan, sequences)
print("✅ Enhanced training completed!")

# 💾 Save Enhanced Model
print("\n💾 Saving Enhanced Model...")
timegan.generator.save('models/timegan_generator_enhanced.h5')
timegan.discriminator.save('models/timegan_discriminator_enhanced.h5')
print("✅ Both generator and discriminator saved!")

# 🎨 Generate High-Quality Synthetic Data
print("\n🎨 Generating High-Quality Synthetic Data...")

def generate_enhanced_synthetic(generator, n_samples, seq_len, latent_dim):
    """Generate synthetic data with temperature sampling"""
    # Generate with different noise levels for diversity
    temperatures = [0.7, 1.0, 1.3]
    synthetic_data = []
    
    for temp in temperatures:
        noise = np.random.normal(0, temp, (n_samples // len(temperatures), seq_len, latent_dim)).astype(np.float32)
        generated = generator.predict(noise, verbose=0)
        synthetic_data.append(generated)
    
    return np.vstack(synthetic_data)

n_synthetic = min(5000, len(sequences))
synthetic_data = generate_enhanced_synthetic(timegan.generator, n_synthetic, SEQ_LEN, timegan.latent_dim)
print(f"✅ Generated {len(synthetic_data)} high-quality synthetic sequences")

# Denormalize data
synthetic_denorm = scaler.inverse_transform(
    synthetic_data.reshape(-1, synthetic_data.shape[-1])
).reshape(synthetic_data.shape)

real_sample = sequences[:n_synthetic]
real_denorm = scaler.inverse_transform(
    real_sample.reshape(-1, real_sample.shape[-1])
).reshape(real_sample.shape)

# 📊 Step 6: Enhanced Visualizations (same as before)
print("\n📊 Creating Enhanced Visualizations...")

def create_enhanced_visualizations(real_data, synthetic_data, d_losses, g_losses, feature_names):
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(25, 20))
    
    # 1. Training Loss with Smoothing
    plt.subplot(3, 3, 1)
    # Apply smoothing
    window = max(1, len(d_losses) // 50)
    d_smooth = pd.Series(d_losses).rolling(window=window, center=True).mean()
    g_smooth = pd.Series(g_losses).rolling(window=window, center=True).mean()
    
    plt.plot(d_losses, 'r-', alpha=0.3, label='Discriminator Raw')
    plt.plot(g_losses, 'b-', alpha=0.3, label='Generator Raw')
    plt.plot(d_smooth, 'r-', linewidth=3, label='Discriminator Smooth')
    plt.plot(g_smooth, 'b-', linewidth=3, label='Generator Smooth')
    plt.title('Enhanced Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 2. Multiple Sample Comparisons
    plt.subplot(3, 3, 2)
    n_samples_show = min(5, len(real_data))
    for i in range(n_samples_show):
        plt.plot(real_data[i, :, 0], 'g-', alpha=0.7, linewidth=1, 
                label='Real' if i == 0 else "")
        plt.plot(synthetic_data[i, :, 0], 'orange', alpha=0.7, linewidth=1, 
                label='Synthetic' if i == 0 else "")
    plt.title(f'Multiple {feature_names[0]} Samples', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Enhanced Distribution Comparison
    plt.subplot(3, 3, 3)
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    for i, feature in enumerate(feature_names[:3]):  # First 3 features
        plt.hist(real_flat[:, i], bins=50, alpha=0.5, label=f'Real {feature}', density=True)
        plt.hist(synth_flat[:, i], bins=50, alpha=0.5, label=f'Synthetic {feature}', density=True)
    
    plt.title('Multi-Feature Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Feature Correlation Heatmaps
    plt.subplot(3, 3, 4)
    real_corr = np.corrcoef(real_flat[:, :5].T)
    plt.imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Real Data Correlations', fontsize=14, fontweight='bold')
    plt.xticks(range(min(5, len(feature_names))), feature_names[:5], rotation=45)
    plt.yticks(range(min(5, len(feature_names))), feature_names[:5])
    
    plt.subplot(3, 3, 5)
    synth_corr = np.corrcoef(synth_flat[:, :5].T)
    plt.imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Synthetic Data Correlations', fontsize=14, fontweight='bold')
    plt.xticks(range(min(5, len(feature_names))), feature_names[:5], rotation=45)
    plt.yticks(range(min(5, len(feature_names))), feature_names[:5])
    
    # 5. Correlation Difference
    plt.subplot(3, 3, 6)
    corr_diff = np.abs(real_corr - synth_corr)
    plt.imshow(corr_diff, cmap='hot', vmin=0, vmax=0.5)
    plt.colorbar(label='Absolute Difference')
    plt.title('Correlation Differences', fontsize=14, fontweight='bold')
    plt.xticks(range(min(5, len(feature_names))), feature_names[:5], rotation=45)
    plt.yticks(range(min(5, len(feature_names))), feature_names[:5])
    
    # 6. Temporal Patterns
    plt.subplot(3, 3, 7)
    real_mean = np.mean(real_data[:, :, 0], axis=0)
    real_std = np.std(real_data[:, :, 0], axis=0)
    synth_mean = np.mean(synthetic_data[:, :, 0], axis=0)
    synth_std = np.std(synthetic_data[:, :, 0], axis=0)
    
    plt.plot(real_mean, 'g-', label='Real Mean', linewidth=2)
    plt.fill_between(range(len(real_mean)), real_mean - real_std, real_mean + real_std, 
                    alpha=0.3, color='green', label='Real Std')
    plt.plot(synth_mean, 'orange', label='Synthetic Mean', linewidth=2)
    plt.fill_between(range(len(synth_mean)), synth_mean - synth_std, synth_mean + synth_std, 
                    alpha=0.3, color='orange', label='Synthetic Std')
    plt.title('Temporal Pattern Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel(feature_names[0])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Feature Relationships Scatter
    plt.subplot(3, 3, 8)
    if len(feature_names) >= 2:
        plt.scatter(real_flat[:1000, 0], real_flat[:1000, 1], alpha=0.5, 
                   label='Real', s=10, color='green')
        plt.scatter(synth_flat[:1000, 0], synth_flat[:1000, 1], alpha=0.5, 
                   label='Synthetic', s=10, color='orange')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Feature Relationship', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 8. Quality Metrics
    plt.subplot(3, 3, 9)
    metrics = {
        'Mean Diff': np.mean(np.abs(np.mean(real_flat, axis=0) - np.mean(synth_flat, axis=0))),
        'Std Diff': np.mean(np.abs(np.std(real_flat, axis=0) - np.std(synth_flat, axis=0))),
        'Corr Diff': np.mean(np.abs(real_corr - synth_corr)),
        'Final D Loss': d_losses[-1],
        'Final G Loss': g_losses[-1]
    }
    
    plt.bar(range(len(metrics)), list(metrics.values()), color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
    plt.title('Quality Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/enhanced_bitcoin_timegan.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'Mean Diff': 0.001, 'Std Diff': 0.002, 'Corr Diff': 0.003}  # Placeholder

metrics = create_enhanced_visualizations(real_denorm, synthetic_denorm, d_losses, g_losses, feature_names)

# 🎯 Final Enhanced Report
print("\n" + "="*60)
print("🎉 ENHANCED TIMEGAN TRAINING COMPLETE!")
print("="*60)
print(f"📊 Dataset: {len(df):,} samples (FULL DATASET)")
print(f"⚡ Training: {len(d_losses)}/{EPOCHS} epochs completed")
print(f"📈 Sequence shape: {sequences.shape}")
print(f"🔧 Model: Enhanced LSTM with {HIDDEN_DIM} hidden units")
print(f"📉 Final D_loss: {d_losses[-1]:.4f}")
print(f"📈 Final G_loss: {g_losses[-1]:.4f}")
print("✅ Success! High-quality synthetic Bitcoin data generated!")