# import os
# import pickle
# import argparse
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, Model, optimizers
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# import warnings
# warnings.filterwarnings('ignore')

# print("🚀 BITCOIN TIMEGAN - FULL DATASET IMPLEMENTATION")
# print("=" * 70)

# # ==================== DATA PREPROCESSING ====================

# def preprocess_bitcoin_data(input_path, out_dir, seq_len=24, stride=6, sample_size=None):
#     """Preprocess FULL Bitcoin dataset for TimeGAN training"""
#     os.makedirs(out_dir, exist_ok=True)

#     print("📊 Reading FULL Bitcoin dataset...")
    
#     # Load data efficiently
#     if sample_size:
#         print(f"🎯 Sampling {sample_size:,} rows for testing...")
#         df = pd.read_csv(input_path, nrows=sample_size)
#     else:
#         print("✅ Loading FULL dataset...")
#         df = pd.read_csv(input_path)
    
#     print(f"📁 Original dataset shape: {df.shape}")
    
#     # Standardize column names
#     column_mapping = {
#         'open': 'Open', 'high': 'High', 'low': 'Low', 
#         'close': 'Close', 'volume': 'Volume',
#         'Open': 'Open', 'High': 'High', 'Low': 'Low', 
#         'Close': 'Close', 'Volume': 'Volume'
#     }
    
#     df.columns = [column_mapping.get(str(col).lower(), col) for col in df.columns]
    
#     # Select Bitcoin columns
#     bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
#     available_cols = [col for col in bitcoin_cols if col in df.columns]
    
#     if not available_cols:
#         numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#         available_cols = numeric_cols[:5]
#         print(f"⚠️ Using numeric columns: {available_cols}")
#     else:
#         print(f"✅ Using Bitcoin columns: {available_cols}")
    
#     df = df[available_cols].copy()
    
#     # Enhanced data cleaning for large dataset
#     print("🧹 Cleaning data...")
#     df = df.replace([np.inf, -np.inf], np.nan)
#     df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
#     df = df[np.isfinite(df).all(axis=1)]
    
#     print(f"✅ Cleaned dataset shape: {df.shape}")
#     print(f"📊 Data sample:")
#     print(df.head(3))
    
#     # Convert to numpy
#     data = df.values.astype(np.float32)
#     feature_names = df.columns.tolist()
#     print("📋 Features:", feature_names)

#     # Create sequences with memory efficiency
#     def make_windows(arr, seq_len, stride):
#         N, D = arr.shape
#         windows = []
#         # Use step size to avoid memory issues with large datasets
#         step_size = max(1, (N - seq_len) // 10000) if N > 10000 else stride
#         for start in range(0, N - seq_len + 1, step_size):
#             windows.append(arr[start:start + seq_len])
#             if len(windows) >= 10000:  # Limit sequences for memory
#                 break
#         return np.stack(windows)

#     print("⏱️ Creating sequences...")
#     windows = make_windows(data, seq_len, stride)
#     print("🪟 Windowed shape:", windows.shape)

#     # Train/Val/Test split
#     n = windows.shape[0]
#     n_train = int(0.7 * n)
#     n_val = int(0.15 * n)
    
#     train, val, test = (
#         windows[:n_train],
#         windows[n_train:n_train + n_val],
#         windows[n_train + n_val:],
#     )

#     # Scaling
#     print("⚖️ Scaling data...")
#     scaler = StandardScaler()
#     D = train.shape[2]
#     scaler.fit(train.reshape(-1, D))
    
#     def scale(x):
#         original_shape = x.shape
#         scaled = scaler.transform(x.reshape(-1, D))
#         return scaled.reshape(original_shape)

#     train_s, val_s, test_s = scale(train), scale(val), scale(test)

#     # Save processed data
#     print("💾 Saving processed data...")
#     np.save(os.path.join(out_dir, "train.npy"), train_s)
#     np.save(os.path.join(out_dir, "val.npy"), val_s)
#     np.save(os.path.join(out_dir, "test.npy"), test_s)
    
#     with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)
        
#     with open(os.path.join(out_dir, "features.txt"), "w") as f:
#         f.write("\n".join(feature_names))

#     print("✅ Saved processed data to", out_dir)
#     print("📊 Final shapes - Train:", train_s.shape, "Val:", val_s.shape, "Test:", test_s.shape)
    
#     return train_s, val_s, test_s, scaler, feature_names

# # ==================== TIMEGAN MODEL COMPONENTS ====================

# class Embedder(Model):
#     def __init__(self, input_dim, hidden_dim=64, num_layers=2):
#         super(Embedder, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.lstm_layers = []
#         for i in range(num_layers):
#             return_sequences = True
#             self.lstm_layers.append(
#                 layers.LSTM(hidden_dim, return_sequences=return_sequences, 
#                            dropout=0.1, recurrent_dropout=0.05)
#             )
        
#     def call(self, x, training=False):
#         for lstm in self.lstm_layers:
#             x = lstm(x, training=training)
#         return x

# class Recovery(Model):
#     def __init__(self, hidden_dim=64, output_dim=5, num_layers=2):
#         super(Recovery, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.num_layers = num_layers
        
#         self.lstm_layers = []
#         for i in range(num_layers):
#             return_sequences = True
#             self.lstm_layers.append(
#                 layers.LSTM(hidden_dim, return_sequences=return_sequences, 
#                            dropout=0.1, recurrent_dropout=0.05)
#             )
        
#         self.output_layer = layers.Dense(output_dim, activation='linear')
        
#     def call(self, x, training=False):
#         for lstm in self.lstm_layers:
#             x = lstm(x, training=training)
#         return self.output_layer(x)

# class Generator(Model):
#     def __init__(self, z_dim=32, hidden_dim=64, num_layers=2):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.lstm_layers = []
#         for i in range(num_layers):
#             return_sequences = True
#             self.lstm_layers.append(
#                 layers.LSTM(hidden_dim, return_sequences=return_sequences, 
#                            dropout=0.1, recurrent_dropout=0.05)
#             )
        
#         self.output_layer = layers.Dense(hidden_dim, activation='tanh')
        
#     def call(self, x, training=False):
#         for lstm in self.lstm_layers:
#             x = lstm(x, training=training)
#         return self.output_layer(x)

# class Supervisor(Model):
#     def __init__(self, hidden_dim=64, num_layers=1):
#         super(Supervisor, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.lstm_layers = []
#         for i in range(num_layers):
#             return_sequences = True
#             self.lstm_layers.append(
#                 layers.LSTM(hidden_dim, return_sequences=return_sequences, 
#                            dropout=0.1, recurrent_dropout=0.05)
#             )
        
#     def call(self, x, training=False):
#         for lstm in self.lstm_layers:
#             x = lstm(x, training=training)
#         return x

# class Discriminator(Model):
#     def __init__(self, hidden_dim=64, num_layers=1):
#         super(Discriminator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.lstm_layers = []
#         for i in range(num_layers):
#             return_sequences = False if i == num_layers - 1 else True
#             self.lstm_layers.append(
#                 layers.LSTM(hidden_dim, return_sequences=return_sequences, 
#                            dropout=0.1, recurrent_dropout=0.05)
#             )
        
#         self.output_layer = layers.Dense(1, activation='sigmoid')
        
#     def call(self, x, training=False):
#         for lstm in self.lstm_layers:
#             x = lstm(x, training=training)
#         return self.output_layer(x)

# # ==================== TRAINING ====================

# class TimeGANTrainer:
#     def __init__(self, seq_len=24, feature_dim=5, hidden_dim=64, z_dim=32):
#         self.seq_len = seq_len
#         self.feature_dim = feature_dim
#         self.hidden_dim = hidden_dim
#         self.z_dim = z_dim
        
#         # Build models
#         self.embedder = Embedder(input_dim=feature_dim, hidden_dim=hidden_dim)
#         self.recovery = Recovery(hidden_dim=hidden_dim, output_dim=feature_dim)
#         self.generator = Generator(z_dim=z_dim, hidden_dim=hidden_dim)
#         self.supervisor = Supervisor(hidden_dim=hidden_dim)
#         self.discriminator = Discriminator(hidden_dim=hidden_dim)
        
#         # Build with dummy data
#         _ = self.embedder(tf.zeros([1, seq_len, feature_dim]))
#         _ = self.recovery(tf.zeros([1, seq_len, hidden_dim]))
#         _ = self.generator(tf.zeros([1, seq_len, z_dim]))
#         _ = self.supervisor(tf.zeros([1, seq_len, hidden_dim]))
#         _ = self.discriminator(tf.zeros([1, seq_len, hidden_dim]))
        
#         # Optimizers
#         self.lr = 1e-4
#         self.opt_e = optimizers.Adam(self.lr)
#         self.opt_r = optimizers.Adam(self.lr)
#         self.opt_g = optimizers.Adam(self.lr)
#         self.opt_s = optimizers.Adam(self.lr)
#         self.opt_d = optimizers.Adam(self.lr)
        
#         # Loss functions
#         self.mse = tf.keras.losses.MeanSquaredError()
#         self.bce = tf.keras.losses.BinaryCrossentropy()
        
#         print("✅ TimeGAN models built successfully!")
#         print(f"   Embedder: {self.embedder.count_params():,} params")
#         print(f"   Recovery: {self.recovery.count_params():,} params")
#         print(f"   Generator: {self.generator.count_params():,} params")
#         print(f"   Supervisor: {self.supervisor.count_params():,} params")
#         print(f"   Discriminator: {self.discriminator.count_params():,} params")
    
#     def sample_z(self, batch_size):
#         return tf.random.normal([batch_size, self.seq_len, self.z_dim])
    
#     @tf.function
#     def train_embedder_recovery(self, x):
#         with tf.GradientTape(persistent=True) as tape:
#             H = self.embedder(x, training=True)
#             X_tilde = self.recovery(H, training=True)
#             loss = self.mse(x, X_tilde)
        
#         # Calculate gradients separately for each model
#         embedder_grads = tape.gradient(loss, self.embedder.trainable_variables)
#         recovery_grads = tape.gradient(loss, self.recovery.trainable_variables)
        
#         # Apply gradients separately
#         self.opt_e.apply_gradients(zip(embedder_grads, self.embedder.trainable_variables))
#         self.opt_r.apply_gradients(zip(recovery_grads, self.recovery.trainable_variables))
        
#         del tape  # Important: delete persistent tape
#         return loss
    
#     @tf.function
#     def train_supervisor(self, x):
#         with tf.GradientTape() as tape:
#             H = self.embedder(x, training=False)
#             H_hat = self.supervisor(H, training=True)
#             loss = self.mse(H[:, 1:, :], H_hat[:, :-1, :])
        
#         grads = tape.gradient(loss, self.supervisor.trainable_variables)
#         self.opt_s.apply_gradients(zip(grads, self.supervisor.trainable_variables))
        
#         return loss
    
#     @tf.function
#     def train_generator(self, x):
#         batch_size = tf.shape(x)[0]
#         z = self.sample_z(batch_size)
        
#         with tf.GradientTape() as tape:
#             # Generate synthetic data
#             E_hat = self.generator(z, training=True)
#             H_hat = self.supervisor(E_hat, training=True)
#             X_hat = self.recovery(H_hat, training=True)
            
#             # Discriminator output
#             Y_fake = self.discriminator(H_hat, training=False)
            
#             # Generator losses
#             g_adv_loss = self.bce(tf.ones_like(Y_fake), Y_fake)
#             g_supervised_loss = self.mse(
#                 self.embedder(x, training=False)[:, 1:, :], 
#                 H_hat[:, :-1, :]
#             )
            
#             # Total loss
#             g_loss = g_adv_loss + 100 * g_supervised_loss
        
#         vars_ = (self.generator.trainable_variables + 
#                 self.supervisor.trainable_variables)
#         grads = tape.gradient(g_loss, vars_)
#         self.opt_g.apply_gradients(zip(grads, vars_))
        
#         return g_loss, g_adv_loss, g_supervised_loss
    
#     @tf.function
#     def train_discriminator(self, x):
#         batch_size = tf.shape(x)[0]
#         z = self.sample_z(batch_size)
        
#         with tf.GradientTape() as tape:
#             # Real data
#             H_real = self.embedder(x, training=False)
#             Y_real = self.discriminator(H_real, training=True)
            
#             # Fake data
#             E_hat = self.generator(z, training=False)
#             H_fake = self.supervisor(E_hat, training=False)
#             Y_fake = self.discriminator(H_fake, training=True)
            
#             # Discriminator losses
#             d_loss_real = self.bce(tf.ones_like(Y_real), Y_real)
#             d_loss_fake = self.bce(tf.zeros_like(Y_fake), Y_fake)
#             d_loss = (d_loss_real + d_loss_fake) * 0.5
        
#         grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
#         self.opt_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
#         return d_loss

#     def train(self, train_data, val_data, epochs=100, batch_size=32):
#         print("🚀 Starting TimeGAN training...")
        
#         # Prepare datasets
#         train_ds = tf.data.Dataset.from_tensor_slices(train_data)
#         train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
#         val_ds = tf.data.Dataset.from_tensor_slices(val_data)
#         val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
#         # Training history
#         history = {
#             'embedder_loss': [], 'supervisor_loss': [],
#             'generator_loss': [], 'discriminator_loss': [],
#             'g_adv_loss': [], 'g_sup_loss': []
#         }
        
#         # Phase 1: Train embedder and recovery
#         print("\n🔧 Phase 1: Training Embedder & Recovery (20 epochs)...")
#         for epoch in range(20):
#             losses = []
#             for batch in train_ds:
#                 loss = self.train_embedder_recovery(batch)
#                 losses.append(loss.numpy())
#             avg_loss = np.mean(losses)
#             history['embedder_loss'].append(avg_loss)
#             if epoch % 5 == 0:
#                 print(f"   Epoch {epoch+1}/20 - Loss: {avg_loss:.6f}")
        
#         # Phase 2: Train supervisor
#         print("\n🔧 Phase 2: Training Supervisor (20 epochs)...")
#         for epoch in range(20):
#             losses = []
#             for batch in train_ds:
#                 loss = self.train_supervisor(batch)
#                 losses.append(loss.numpy())
#             avg_loss = np.mean(losses)
#             history['supervisor_loss'].append(avg_loss)
#             if epoch % 5 == 0:
#                 print(f"   Epoch {epoch+1}/20 - Loss: {avg_loss:.6f}")
        
#         # Phase 3: Joint training
#         print(f"\n🔧 Phase 3: Joint Adversarial Training ({epochs} epochs)...")
#         for epoch in range(epochs):
#             # Train discriminator
#             d_losses = []
#             for batch in train_ds:
#                 d_loss = self.train_discriminator(batch)
#                 d_losses.append(d_loss.numpy())
            
#             # Train generator
#             g_losses, g_adv_losses, g_sup_losses = [], [], []
#             for batch in train_ds:
#                 g_loss, g_adv, g_sup = self.train_generator(batch)
#                 g_losses.append(g_loss.numpy())
#                 g_adv_losses.append(g_adv.numpy())
#                 g_sup_losses.append(g_sup.numpy())
            
#             # Record history
#             avg_d_loss = np.mean(d_losses)
#             avg_g_loss = np.mean(g_losses)
#             avg_g_adv = np.mean(g_adv_losses)
#             avg_g_sup = np.mean(g_sup_losses)
            
#             history['discriminator_loss'].append(avg_d_loss)
#             history['generator_loss'].append(avg_g_loss)
#             history['g_adv_loss'].append(avg_g_adv)
#             history['g_sup_loss'].append(avg_g_sup)
            
#             if epoch % 10 == 0 or epoch == epochs - 1:
#                 print(f"   Epoch {epoch+1}/{epochs} - "
#                       f"D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}, "
#                       f"G_adv: {avg_g_adv:.4f}, G_sup: {avg_g_sup:.4f}")
        
#         print("✅ Training completed!")
#         return history

#     def generate_synthetic_data(self, n_samples, scaler, batch_size=1000):
#         """Generate synthetic data in batches to avoid memory issues"""
#         print(f"🎨 Generating {n_samples} synthetic samples...")
        
#         synthetic_data = []
#         n_batches = (n_samples + batch_size - 1) // batch_size
        
#         for i in range(n_batches):
#             current_batch = min(batch_size, n_samples - i * batch_size)
#             z = tf.random.normal([current_batch, self.seq_len, self.z_dim])
            
#             E_hat = self.generator(z, training=False)
#             H_hat = self.supervisor(E_hat, training=False)
#             X_hat = self.recovery(H_hat, training=False).numpy()
            
#             synthetic_data.append(X_hat)
            
#             if (i + 1) % 5 == 0:
#                 print(f"   Generated batch {i+1}/{n_batches}")
        
#         synthetic_data = np.vstack(synthetic_data)
        
#         # Inverse transform
#         n, T, D = synthetic_data.shape
#         print(f"🔄 Inverse scaling {synthetic_data.shape} data...")
#         synthetic_inv = scaler.inverse_transform(synthetic_data.reshape(-1, D)).reshape(n, T, D)
        
#         return synthetic_inv

# # ==================== EVALUATION ====================

# def evaluate_synthetic_data(real_data, synthetic_data, n_samples=1000):
#     """Evaluate quality of synthetic data"""
#     print("📊 Evaluating synthetic data quality...")
    
#     # Sample subsets for efficiency
#     n_eval = min(n_samples, real_data.shape[0], synthetic_data.shape[0])
#     real_sample = real_data[:n_eval]
#     synth_sample = synthetic_data[:n_eval]
    
#     # Flatten for some metrics
#     real_flat = real_sample.reshape(real_sample.shape[0], -1)
#     synth_flat = synth_sample.reshape(synth_sample.shape[0], -1)
    
#     metrics = {}
    
#     # 1. Basic statistics comparison
#     real_mean = np.mean(real_flat, axis=0)
#     synth_mean = np.mean(synth_flat, axis=0)
#     real_std = np.std(real_flat, axis=0)
#     synth_std = np.std(synth_flat, axis=0)
    
#     metrics['mean_difference'] = np.mean(np.abs(real_mean - synth_mean))
#     metrics['std_difference'] = np.mean(np.abs(real_std - synth_std))
    
#     # 2. Correlation comparison
#     real_corr = np.corrcoef(real_flat.T)
#     synth_corr = np.corrcoef(synth_flat.T)
#     metrics['correlation_difference'] = np.mean(np.abs(real_corr - synth_corr))
    
#     # 3. MMD (Maximum Mean Discrepancy)
#     def mmd_rbf(X, Y, sigma=None):
#         X = X.reshape(X.shape[0], -1)
#         Y = Y.reshape(Y.shape[0], -1)
        
#         if sigma is None:
#             # Median heuristic
#             XX = cdist(X, X, 'euclidean')
#             sigma = np.median(XX[XX > 0]) if np.any(XX > 0) else 1.0
        
#         K_XX = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * sigma**2))
#         K_YY = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * sigma**2))
#         K_XY = np.exp(-cdist(X, Y, 'sqeuclidean') / (2 * sigma**2))
        
#         mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
#         return mmd
    
#     metrics['mmd'] = mmd_rbf(real_sample, synth_sample)
    
#     print("📈 Evaluation Results:")
#     print(f"   ✅ Mean Difference: {metrics['mean_difference']:.6f}")
#     print(f"   ✅ Std Difference: {metrics['std_difference']:.6f}")
#     print(f"   ✅ Correlation Difference: {metrics['correlation_difference']:.6f}")
#     print(f"   ✅ MMD: {metrics['mmd']:.6f}")
    
#     return metrics

# def plot_results(real_data, synthetic_data, history, feature_names, save_path='timegan_results.png'):
#     """Plot training results and data comparisons"""
#     print("📊 Generating plots...")
    
#     fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
#     # 1. Training losses
#     if history['embedder_loss']:
#         axes[0,0].plot(history['embedder_loss'], label='Embedder', alpha=0.7)
#     if history['supervisor_loss']:
#         axes[0,0].plot(history['supervisor_loss'], label='Supervisor', alpha=0.7)
#     if history['discriminator_loss']:
#         axes[0,0].plot(history['discriminator_loss'], label='Discriminator', alpha=0.7)
#     if history['generator_loss']:
#         axes[0,0].plot(history['generator_loss'], label='Generator', alpha=0.7)
#     axes[0,0].set_title('Training Losses')
#     axes[0,0].set_xlabel('Epoch')
#     axes[0,0].set_ylabel('Loss')
#     axes[0,0].legend()
#     axes[0,0].grid(True, alpha=0.3)
    
#     # 2. Sample comparison
#     n_show = min(5, real_data.shape[0])
#     time_steps = range(real_data.shape[1])
    
#     for i in range(n_show):
#         axes[0,1].plot(time_steps, real_data[i, :, 0], 'g-', alpha=0.5, label='Real' if i == 0 else "")
#         axes[0,1].plot(time_steps, synthetic_data[i, :, 0], 'orange', alpha=0.5, label='Synthetic' if i == 0 else "")
#     axes[0,1].set_title(f'Sample Comparison - {feature_names[0]}')
#     axes[0,1].set_xlabel('Time Steps')
#     axes[0,1].set_ylabel('Value')
#     axes[0,1].legend()
#     axes[0,1].grid(True, alpha=0.3)
    
#     # 3. Distribution comparison
#     real_flat = real_data.reshape(-1, real_data.shape[-1])
#     synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
#     axes[0,2].hist(real_flat[:, 0], bins=50, alpha=0.6, label='Real', density=True, color='green')
#     axes[0,2].hist(synth_flat[:, 0], bins=50, alpha=0.6, label='Synthetic', density=True, color='orange')
#     axes[0,2].set_title('Distribution Comparison')
#     axes[0,2].set_xlabel('Value')
#     axes[0,2].set_ylabel('Density')
#     axes[0,2].legend()
#     axes[0,2].grid(True, alpha=0.3)
    
#     # 4. Correlation matrices
#     real_corr = np.corrcoef(real_flat[:, :5].T)
#     synth_corr = np.corrcoef(synth_flat[:, :5].T)
    
#     im1 = axes[1,0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
#     axes[1,0].set_title('Real Data Correlations')
#     axes[1,0].set_xticks(range(len(feature_names[:5])))
#     axes[1,0].set_xticklabels(feature_names[:5], rotation=45)
#     axes[1,0].set_yticks(range(len(feature_names[:5])))
#     axes[1,0].set_yticklabels(feature_names[:5])
#     plt.colorbar(im1, ax=axes[1,0])
    
#     im2 = axes[1,1].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1)
#     axes[1,1].set_title('Synthetic Data Correlations')
#     axes[1,1].set_xticks(range(len(feature_names[:5])))
#     axes[1,1].set_xticklabels(feature_names[:5], rotation=45)
#     axes[1,1].set_yticks(range(len(feature_names[:5])))
#     axes[1,1].set_yticklabels(feature_names[:5])
#     plt.colorbar(im2, ax=axes[1,1])
    
#     # 5. Correlation difference
#     corr_diff = np.abs(real_corr - synth_corr)
#     im3 = axes[1,2].imshow(corr_diff, cmap='hot', vmin=0, vmax=0.5)
#     axes[1,2].set_title('Correlation Differences')
#     axes[1,2].set_xticks(range(len(feature_names[:5])))
#     axes[1,2].set_xticklabels(feature_names[:5], rotation=45)
#     axes[1,2].set_yticks(range(len(feature_names[:5])))
#     axes[1,2].set_yticklabels(feature_names[:5])
#     plt.colorbar(im3, ax=axes[1,2])
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"💾 Plot saved as: {save_path}")
#     plt.show()

# # ==================== MAIN EXECUTION ====================

# def main():
#     parser = argparse.ArgumentParser(description='TimeGAN for FULL Bitcoin Dataset')
#     parser.add_argument('--input', type=str, required=True, help='Path to Bitcoin CSV file')
#     parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
#     parser.add_argument('--seq_len', type=int, default=24, help='Sequence length')
#     parser.add_argument('--n_samples', type=int, default=5000, help='Number of synthetic samples to generate')
#     parser.add_argument('--sample_size', type=int, default=None, help='Sample size for testing (None for full dataset)')
    
#     args = parser.parse_args()
    
#     # Create output directories
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'synthetic'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'processed'), exist_ok=True)
    
#     print("=" * 70)
#     print("🚀 BITCOIN TIMEGAN - FULL DATASET PIPELINE")
#     print("=" * 70)
    
#     # Step 1: Preprocess data
#     print("\n📊 STEP 1: Preprocessing data...")
#     train_data, val_data, test_data, scaler, feature_names = preprocess_bitcoin_data(
#         args.input, 
#         os.path.join(args.output_dir, 'processed'),
#         seq_len=args.seq_len,
#         sample_size=args.sample_size
#     )
    
#     # Step 2: Initialize and train TimeGAN
#     print("\n🧠 STEP 2: Building and training TimeGAN...")
#     trainer = TimeGANTrainer(
#         seq_len=args.seq_len,
#         feature_dim=train_data.shape[2],
#         hidden_dim=64,
#         z_dim=32
#     )
    
#     history = trainer.train(
#         train_data, val_data, 
#         epochs=args.epochs, 
#         batch_size=args.batch_size
#     )
    
#     # Step 3: Generate synthetic data
#     print("\n🎨 STEP 3: Generating synthetic data...")
#     synthetic_data = trainer.generate_synthetic_data(args.n_samples, scaler)
    
#     # Step 4: Evaluate synthetic data
#     print("\n📈 STEP 4: Evaluating synthetic data...")
#     real_test_denorm = scaler.inverse_transform(
#         test_data.reshape(-1, test_data.shape[-1])
#     ).reshape(test_data.shape)
    
#     metrics = evaluate_synthetic_data(real_test_denorm, synthetic_data)
    
#     # Step 5: Save results
#     print("\n💾 STEP 5: Saving results...")
    
#     # Save synthetic data
#     synthetic_path = os.path.join(args.output_dir, 'synthetic', 'synthetic_bitcoin.npy')
#     np.save(synthetic_path, synthetic_data)
    
#     csv_path = os.path.join(args.output_dir, 'synthetic', 'synthetic_bitcoin.csv')
#     np.savetxt(csv_path, synthetic_data.reshape(synthetic_data.shape[0], -1), delimiter=',')
    
#     # # Save models
#     # trainer.embedder.save_weights(os.path.join(args.output_dir, 'models', 'embedder.h5'))
#     # trainer.recovery.save_weights(os.path.join(args.output_dir, 'models', 'recovery.h5'))
#     # trainer.generator.save_weights(os.path.join(args.output_dir, 'models', 'generator.h5'))
#     # trainer.supervisor.save_weights(os.path.join(args.output_dir, 'models', 'supervisor.h5'))
#     # trainer.discriminator.save_weights(os.path.join(args.output_dir, 'models', 'discriminator.h5'))
#     # In the main() function, replace the model saving section with:

#     # Save models with correct extension
#     trainer.embedder.save_weights(os.path.join(args.output_dir, 'models', 'embedder.weights.h5'))
#     trainer.recovery.save_weights(os.path.join(args.output_dir, 'models', 'recovery.weights.h5'))
#     trainer.generator.save_weights(os.path.join(args.output_dir, 'models', 'generator.weights.h5'))
#     trainer.supervisor.save_weights(os.path.join(args.output_dir, 'models', 'supervisor.weights.h5'))
#     trainer.discriminator.save_weights(os.path.join(args.output_dir, 'models', 'discriminator.weights.h5'))
    
#     # Save metrics and history
#     metrics_path = os.path.join(args.output_dir, 'synthetic', 'evaluation_metrics.json')
    
#     # Convert numpy types to Python native types for JSON serialization
#     metrics_serializable = {}
#     for key, value in metrics.items():
#         if hasattr(value, 'item'):  # For numpy types
#             metrics_serializable[key] = value.item()
#         else:
#             metrics_serializable[key] = value
    
#     with open(metrics_path, 'w') as f:
#         import json
#         json.dump(metrics_serializable, f, indent=2)
    
#     history_path = os.path.join(args.output_dir, 'synthetic', 'training_history.json')
    
#     # Also convert history values to native Python types
#     history_serializable = {}
#     for key, values in history.items():
#         history_serializable[key] = [float(v) if hasattr(v, 'item') else float(v) for v in values]
    
#     with open(history_path, 'w') as f:
#         json.dump(history_serializable, f, indent=2) 
#     # Step 6: Plot results
#     print("\n📊 STEP 6: Generating plots...")
#     plot_path = os.path.join(args.output_dir, 'timegan_results.png')
#     plot_results(real_test_denorm[:1000], synthetic_data[:1000], history, feature_names, plot_path)
    
#     # Final report
#     print("\n" + "=" * 70)
#     print("🎉 TIMEGAN PIPELINE COMPLETED SUCCESSFULLY!")
#     print("=" * 70)
#     print(f"📊 Real data shape: {real_test_denorm.shape}")
#     print(f"🎨 Synthetic data shape: {synthetic_data.shape}")
#     print(f"📈 Quality metrics: {metrics_path}")
#     print(f"📊 Training history: {history_path}")
#     print(f"💾 Synthetic data: {synthetic_path}")
#     print(f"💾 Models: {os.path.join(args.output_dir, 'models')}")
#     print(f"🖼️  Results plot: {plot_path}")
    
#     # Quality assessment
#     if metrics['mean_difference'] < 0.1 and metrics['correlation_difference'] < 0.2:
#         print("🎉 EXCELLENT: High-quality synthetic data generated!")
#     elif metrics['mean_difference'] < 0.2 and metrics['correlation_difference'] < 0.3:
#         print("✅ GOOD: Synthetic data generated with acceptable quality")
#     else:
#         print("⚠️ FAIR: Synthetic data generated but may need more training")
    
#     print("=" * 70)

# if __name__ == "__main__":
#     main()



















































import os
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

print("🚀 BITCOIN TIMEGAN - IMPROVED IMPLEMENTATION")
print("=" * 70)

# ==================== IMPROVED DATA PREPROCESSING ====================

def preprocess_bitcoin_data_improved(input_path, out_dir, seq_len=24, stride=6, sample_size=500000):
    """Improved preprocessing with better data handling"""
    os.makedirs(out_dir, exist_ok=True)

    print("📊 Reading Bitcoin dataset...")
    
    # Load limited data for better training
    if sample_size:
        print(f"🎯 Sampling {sample_size:,} rows for manageable training...")
        df = pd.read_csv(input_path, nrows=sample_size)
    else:
        df = pd.read_csv(input_path)
        # Still sample for reasonable training time
        df = df.sample(min(50000, len(df)), random_state=42)
    
    print(f"📁 Working dataset shape: {df.shape}")
    
    # Standardize column names
    column_mapping = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume'
    }
    
    df.columns = [column_mapping.get(str(col).lower(), col) for col in df.columns]
    
    # Select Bitcoin columns
    bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in bitcoin_cols if col in df.columns]
    
    print(f"✅ Using columns: {available_cols}")
    df = df[available_cols].copy()
    
    # Enhanced data cleaning
    print("🧹 Cleaning data...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers
    for col in available_cols:
        if col != 'Volume':  # Volume can have different distribution
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"✅ Cleaned dataset shape: {df.shape}")
    
    # Convert to numpy
    data = df.values.astype(np.float32)
    feature_names = df.columns.tolist()
    print("📋 Features:", feature_names)

    # Create sequences
    def make_windows(arr, seq_len, stride):
        N, D = arr.shape
        windows = []
        for start in range(0, N - seq_len + 1, stride):
            windows.append(arr[start:start + seq_len])
        return np.stack(windows)

    print("⏱️ Creating sequences...")
    windows = make_windows(data, seq_len, stride)
    print("🪟 Windowed shape:", windows.shape)

    # Train/Val/Test split
    n = windows.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train, val, test = (
        windows[:n_train],
        windows[n_train:n_train + n_val],
        windows[n_train + n_val:],
    )

    # Improved scaling
    print("⚖️ Scaling data...")
    scaler = StandardScaler()
    D = train.shape[2]
    
    # Fit on training data only
    scaler.fit(train.reshape(-1, D))
    
    def scale(x):
        original_shape = x.shape
        scaled = scaler.transform(x.reshape(-1, D))
        return scaled.reshape(original_shape)

    train_s, val_s, test_s = scale(train), scale(val), scale(test)

    # Save processed data
    print("💾 Saving processed data...")
    np.save(os.path.join(out_dir, "train.npy"), train_s)
    np.save(os.path.join(out_dir, "val.npy"), val_s)
    np.save(os.path.join(out_dir, "test.npy"), test_s)
    
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))

    print("✅ Saved processed data to", out_dir)
    print("📊 Final shapes - Train:", train_s.shape, "Val:", val_s.shape, "Test:", test_s.shape)
    
    return train_s, val_s, test_s, scaler, feature_names

# ==================== IMPROVED TIMEGAN MODEL ====================

class ImprovedEmbedder(Model):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(ImprovedEmbedder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Use GRU instead of LSTM for faster training
        self.gru_layers = []
        for i in range(num_layers):
            return_sequences = True
            self.gru_layers.append(
                layers.GRU(hidden_dim, return_sequences=return_sequences, 
                          dropout=0.2, recurrent_dropout=0.1)
            )
        
    def call(self, x, training=False):
        for gru in self.gru_layers:
            x = gru(x, training=training)
        return x

class ImprovedRecovery(Model):
    def __init__(self, hidden_dim=128, output_dim=5, num_layers=3):
        super(ImprovedRecovery, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gru_layers = []
        for i in range(num_layers):
            return_sequences = True
            self.gru_layers.append(
                layers.GRU(hidden_dim, return_sequences=return_sequences, 
                          dropout=0.2, recurrent_dropout=0.1)
            )
        
        self.output_layer = layers.Dense(output_dim, activation='linear')
        
    def call(self, x, training=False):
        for gru in self.gru_layers:
            x = gru(x, training=training)
        return self.output_layer(x)

class ImprovedGenerator(Model):
    def __init__(self, z_dim=64, hidden_dim=128, num_layers=3):
        super(ImprovedGenerator, self).__init__()
        self.z_dim = z_dim
        
        self.gru_layers = []
        for i in range(num_layers):
            return_sequences = True
            self.gru_layers.append(
                layers.GRU(hidden_dim, return_sequences=return_sequences, 
                          dropout=0.2, recurrent_dropout=0.1)
            )
        
        self.output_layer = layers.Dense(hidden_dim, activation='tanh')
        
    def call(self, x, training=False):
        for gru in self.gru_layers:
            x = gru(x, training=training)
        return self.output_layer(x)

class ImprovedSupervisor(Model):
    def __init__(self, hidden_dim=128, num_layers=2):
        super(ImprovedSupervisor, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gru_layers = []
        for i in range(num_layers):
            return_sequences = True
            self.gru_layers.append(
                layers.GRU(hidden_dim, return_sequences=return_sequences, 
                          dropout=0.2, recurrent_dropout=0.1)
            )
        
    def call(self, x, training=False):
        for gru in self.gru_layers:
            x = gru(x, training=training)
        return x

class ImprovedDiscriminator(Model):
    def __init__(self, hidden_dim=128, num_layers=2):
        super(ImprovedDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gru_layers = []
        for i in range(num_layers):
            return_sequences = False if i == num_layers - 1 else True
            self.gru_layers.append(
                layers.GRU(hidden_dim, return_sequences=return_sequences, 
                          dropout=0.2, recurrent_dropout=0.1)
            )
        
        self.output_layer = layers.Dense(1, activation='sigmoid')
        
    def call(self, x, training=False):
        for gru in self.gru_layers:
            x = gru(x, training=training)
        return self.output_layer(x)

# ==================== IMPROVED TRAINING ====================

class ImprovedTimeGANTrainer:
    def __init__(self, seq_len=24, feature_dim=5, hidden_dim=128, z_dim=64):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # Build improved models
        self.embedder = ImprovedEmbedder(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.recovery = ImprovedRecovery(hidden_dim=hidden_dim, output_dim=feature_dim)
        self.generator = ImprovedGenerator(z_dim=z_dim, hidden_dim=hidden_dim)
        self.supervisor = ImprovedSupervisor(hidden_dim=hidden_dim)
        self.discriminator = ImprovedDiscriminator(hidden_dim=hidden_dim)
        
        # Build with dummy data
        _ = self.embedder(tf.zeros([1, seq_len, feature_dim]))
        _ = self.recovery(tf.zeros([1, seq_len, hidden_dim]))
        _ = self.generator(tf.zeros([1, seq_len, z_dim]))
        _ = self.supervisor(tf.zeros([1, seq_len, hidden_dim]))
        _ = self.discriminator(tf.zeros([1, seq_len, hidden_dim]))
        
        # Improved optimizers with different learning rates
        self.opt_e = optimizers.Adam(1e-3)
        self.opt_r = optimizers.Adam(1e-3)
        self.opt_g = optimizers.Adam(5e-4)
        self.opt_s = optimizers.Adam(5e-4)
        self.opt_d = optimizers.Adam(2e-4)  # Slower for discriminator
        
        # Loss functions
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
        
        print("✅ Improved TimeGAN models built successfully!")
        print(f"   Embedder: {self.embedder.count_params():,} params")
        print(f"   Recovery: {self.recovery.count_params():,} params")
        print(f"   Generator: {self.generator.count_params():,} params")
        print(f"   Supervisor: {self.supervisor.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
    
    def sample_z(self, batch_size):
        return tf.random.normal([batch_size, self.seq_len, self.z_dim])
    
    @tf.function
    def train_embedder_recovery(self, x):
        with tf.GradientTape(persistent=True) as tape:
            H = self.embedder(x, training=True)
            X_tilde = self.recovery(H, training=True)
            loss = self.mse(x, X_tilde)
        
        embedder_grads = tape.gradient(loss, self.embedder.trainable_variables)
        recovery_grads = tape.gradient(loss, self.recovery.trainable_variables)
        
        # Apply gradients with gradient clipping
        self.opt_e.apply_gradients(zip(embedder_grads, self.embedder.trainable_variables))
        self.opt_r.apply_gradients(zip(recovery_grads, self.recovery.trainable_variables))
        
        del tape
        return loss
    
    @tf.function
    def train_supervisor(self, x):
        with tf.GradientTape() as tape:
            H = self.embedder(x, training=False)
            H_hat = self.supervisor(H, training=True)
            loss = self.mse(H[:, 1:, :], H_hat[:, :-1, :])
        
        grads = tape.gradient(loss, self.supervisor.trainable_variables)
        self.opt_s.apply_gradients(zip(grads, self.supervisor.trainable_variables))
        
        return loss
    
    @tf.function
    def train_generator(self, x):
        batch_size = tf.shape(x)[0]
        z = self.sample_z(batch_size)
        
        with tf.GradientTape() as tape:
            # Generate synthetic data
            E_hat = self.generator(z, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            X_hat = self.recovery(H_hat, training=True)
            
            # Discriminator output
            Y_fake = self.discriminator(H_hat, training=False)
            
            # Generator losses - improved weighting
            g_adv_loss = self.bce(tf.ones_like(Y_fake), Y_fake)
            g_supervised_loss = self.mse(
                self.embedder(x, training=False)[:, 1:, :], 
                H_hat[:, :-1, :]
            )
            
            # Total loss with better balance
            g_loss = g_adv_loss + 10 * g_supervised_loss  # Reduced from 100
        
        vars_ = (self.generator.trainable_variables + 
                self.supervisor.trainable_variables)
        grads = tape.gradient(g_loss, vars_)
        self.opt_g.apply_gradients(zip(grads, vars_))
        
        return g_loss, g_adv_loss, g_supervised_loss
    
    @tf.function
    def train_discriminator(self, x):
        batch_size = tf.shape(x)[0]
        z = self.sample_z(batch_size)
        
        with tf.GradientTape() as tape:
            # Real data
            H_real = self.embedder(x, training=False)
            Y_real = self.discriminator(H_real, training=True)
            
            # Fake data
            E_hat = self.generator(z, training=False)
            H_fake = self.supervisor(E_hat, training=False)
            Y_fake = self.discriminator(H_fake, training=True)
            
            # Discriminator losses with label smoothing
            d_loss_real = self.bce(tf.ones_like(Y_real) * 0.9, Y_real)  # Label smoothing
            d_loss_fake = self.bce(tf.zeros_like(Y_fake) + 0.1, Y_fake)  # Label smoothing
            d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        return d_loss

    def train(self, train_data, val_data, epochs=200, batch_size=128):
        print("🚀 Starting IMPROVED TimeGAN training...")
        
        # Prepare datasets
        train_ds = tf.data.Dataset.from_tensor_slices(train_data)
        train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Training history
        history = {
            'embedder_loss': [], 'supervisor_loss': [],
            'generator_loss': [], 'discriminator_loss': [],
            'g_adv_loss': [], 'g_sup_loss': []
        }
        
        # Phase 1: Train embedder and recovery (more epochs)
        print("\n🔧 Phase 1: Training Embedder & Recovery (30 epochs)...")
        for epoch in range(30):
            losses = []
            for batch in train_ds:
                loss = self.train_embedder_recovery(batch)
                losses.append(loss.numpy())
            avg_loss = np.mean(losses)
            history['embedder_loss'].append(avg_loss)
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1}/30 - Loss: {avg_loss:.6f}")
        
        # Phase 2: Train supervisor (more epochs)
        print("\n🔧 Phase 2: Training Supervisor (30 epochs)...")
        for epoch in range(30):
            losses = []
            for batch in train_ds:
                loss = self.train_supervisor(batch)
                losses.append(loss.numpy())
            avg_loss = np.mean(losses)
            history['supervisor_loss'].append(avg_loss)
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1}/30 - Loss: {avg_loss:.6f}")
        
        # Phase 3: Joint training with better monitoring
        print(f"\n🔧 Phase 3: Joint Adversarial Training ({epochs} epochs)...")
        best_g_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train discriminator (twice per generator update)
            d_losses = []
            for _ in range(2):
                for batch in train_ds:
                    d_loss = self.train_discriminator(batch)
                    d_losses.append(d_loss.numpy())
                    break  # One batch per update
            
            # Train generator
            g_losses, g_adv_losses, g_sup_losses = [], [], []
            for batch in train_ds:
                g_loss, g_adv, g_sup = self.train_generator(batch)
                g_losses.append(g_loss.numpy())
                g_adv_losses.append(g_adv.numpy())
                g_sup_losses.append(g_sup.numpy())
                break  # One batch per update
            
            # Record history
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            avg_g_adv = np.mean(g_adv_losses)
            avg_g_sup = np.mean(g_sup_losses)
            
            history['discriminator_loss'].append(avg_d_loss)
            history['generator_loss'].append(avg_g_loss)
            history['g_adv_loss'].append(avg_g_adv)
            history['g_sup_loss'].append(avg_g_sup)
            
            # Early stopping check
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch+1}/{epochs} - "
                      f"D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}, "
                      f"G_adv: {avg_g_adv:.4f}, G_sup: {avg_g_sup:.4f}")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        print("✅ Training completed!")
        return history

    def generate_synthetic_data(self, n_samples, scaler, batch_size=1000):
        """Generate synthetic data"""
        print(f"🎨 Generating {n_samples} synthetic samples...")
        
        synthetic_data = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            current_batch = min(batch_size, n_samples - i * batch_size)
            z = tf.random.normal([current_batch, self.seq_len, self.z_dim])
            
            E_hat = self.generator(z, training=False)
            H_hat = self.supervisor(E_hat, training=False)
            X_hat = self.recovery(H_hat, training=False).numpy()
            
            synthetic_data.append(X_hat)
        
        synthetic_data = np.vstack(synthetic_data)
        
        # Inverse transform
        n, T, D = synthetic_data.shape
        print(f"🔄 Inverse scaling {synthetic_data.shape} data...")
        synthetic_inv = scaler.inverse_transform(synthetic_data.reshape(-1, D)).reshape(n, T, D)
        
        return synthetic_inv

# ==================== IMPROVED EVALUATION ====================

def evaluate_synthetic_data_improved(real_data, synthetic_data):
    """Improved evaluation with better metrics"""
    print("📊 Evaluating synthetic data quality...")
    
    # Use smaller subsets for stable evaluation
    n_eval = min(500, real_data.shape[0], synthetic_data.shape[0])
    real_sample = real_data[:n_eval]
    synth_sample = synthetic_data[:n_eval]
    
    # Flatten for metrics
    real_flat = real_sample.reshape(real_sample.shape[0], -1)
    synth_flat = synth_sample.reshape(synth_sample.shape[0], -1)
    
    metrics = {}
    
    # 1. Basic statistics comparison
    real_mean = np.mean(real_flat, axis=0)
    synth_mean = np.mean(synth_flat, axis=0)
    real_std = np.std(real_flat, axis=0)
    synth_std = np.std(synth_flat, axis=0)
    
    metrics['mean_difference'] = np.mean(np.abs(real_mean - synth_mean))
    metrics['std_difference'] = np.mean(np.abs(real_std - synth_std))
    
    # 2. Correlation comparison
    real_corr = np.corrcoef(real_flat.T)
    synth_corr = np.corrcoef(synth_flat.T)
    metrics['correlation_difference'] = np.mean(np.abs(real_corr - synth_corr))
    
    # 3. Distribution similarity using Wasserstein distance
    from scipy.stats import wasserstein_distance
    
    wasserstein_dists = []
    for i in range(min(5, real_flat.shape[1])):
        wasserstein_dists.append(wasserstein_distance(real_flat[:, i], synth_flat[:, i]))
    
    metrics['wasserstein_distance'] = np.mean(wasserstein_dists)
    
    print("📈 IMPROVED Evaluation Results:")
    print(f"   📊 Mean Difference: {metrics['mean_difference']:.2f}")
    print(f"   📊 Std Difference: {metrics['std_difference']:.2f}")
    print(f"   📊 Correlation Difference: {metrics['correlation_difference']:.4f}")
    print(f"   📊 Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
    
    return metrics

# ==================== IMPROVED PLOTTING ====================

def create_comprehensive_analysis(real_data, synthetic_data, history, feature_names, scaler, save_dir='outputs'):
    """Create comprehensive analysis with proper scaling"""
    
    print("📊 Creating comprehensive analysis...")
    
    # Apply inverse scaling for proper visualization
    n_real, T_real, D_real = real_data.shape
    n_synth, T_synth, D_synth = synthetic_data.shape
    
    real_flat = real_data.reshape(-1, D_real)
    synth_flat = synthetic_data.reshape(-1, D_synth)
    
    real_inv = scaler.inverse_transform(real_flat)
    synth_inv = scaler.inverse_transform(synth_flat)
    
    real_data_denorm = real_inv.reshape(n_real, T_real, D_real)
    synthetic_data_denorm = synth_inv.reshape(n_synth, T_synth, D_synth)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Improved TimeGAN - Bitcoin Data Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training history
    if history['generator_loss']:
        axes[0,0].plot(history['generator_loss'], label='Generator', color='red', linewidth=1)
        axes[0,0].plot(history['discriminator_loss'], label='Discriminator', color='blue', linewidth=1)
        axes[0,0].set_title('Training Losses')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Sample comparison
    n_show = min(3, real_data_denorm.shape[0])
    time_steps = range(T_real)
    
    for i in range(n_show):
        axes[0,1].plot(time_steps, real_data_denorm[i, :, 0], 
                      color='green', alpha=0.7, linewidth=2, label='Real' if i == 0 else "")
        axes[0,1].plot(time_steps, synthetic_data_denorm[i, :, 0], 
                      color='orange', alpha=0.7, linewidth=2, linestyle='--', label='Synthetic' if i == 0 else "")
    
    axes[0,1].set_title(f'Sample Comparison - {feature_names[0]}')
    axes[0,1].set_xlabel('Time Steps')
    axes[0,1].set_ylabel('Price ($)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    real_all = real_data_denorm.reshape(-1, D_real)
    synth_all = synthetic_data_denorm.reshape(-1, D_synth)
    
    # Use reasonable price range
    price_range = np.percentile(real_all[:, 0], [1, 99])
    bins = np.linspace(price_range[0], price_range[1], 40)
    
    axes[0,2].hist(real_all[:, 0], bins=bins, alpha=0.7, label='Real', 
                  density=True, color='green', edgecolor='black')
    axes[0,2].hist(synth_all[:, 0], bins=bins, alpha=0.7, label='Synthetic', 
                  density=True, color='orange', edgecolor='black')
    axes[0,2].set_title('Price Distribution')
    axes[0,2].set_xlabel('Price ($)')
    axes[0,2].set_ylabel('Density')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Statistical comparison
    stats_names = ['Mean', 'Std Dev', 'Median']
    real_stats = [np.mean(real_all[:, 0]), np.std(real_all[:, 0]), np.median(real_all[:, 0])]
    synth_stats = [np.mean(synth_all[:, 0]), np.std(synth_all[:, 0]), np.median(synth_all[:, 0])]
    
    x_pos = np.arange(len(stats_names))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, real_stats, width, label='Real', 
                 color='green', alpha=0.7, edgecolor='black')
    axes[1,0].bar(x_pos + width/2, synth_stats, width, label='Synthetic', 
                 color='orange', alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Statistical Comparison')
    axes[1,0].set_xlabel('Statistics')
    axes[1,0].set_ylabel('Price ($)')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(stats_names)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Correlation matrices
    real_corr = np.corrcoef(real_all[:, :3].T)
    synth_corr = np.corrcoef(synth_all[:, :3].T)
    
    im1 = axes[1,1].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,1].set_title('Real Data Correlations')
    axes[1,1].set_xticks(range(3))
    axes[1,1].set_xticklabels(feature_names[:3], rotation=45)
    axes[1,1].set_yticks(range(3))
    axes[1,1].set_yticklabels(feature_names[:3])
    plt.colorbar(im1, ax=axes[1,1])
    
    im2 = axes[1,2].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,2].set_title('Synthetic Data Correlations')
    axes[1,2].set_xticks(range(3))
    axes[1,2].set_xticklabels(feature_names[:3], rotation=45)
    axes[1,2].set_yticks(range(3))
    axes[1,2].set_yticklabels(feature_names[:3])
    plt.colorbar(im2, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improved_timegan_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Analysis saved as: {os.path.join(save_dir, 'improved_timegan_analysis.png')}")

# ==================== MAIN EXECUTION ====================

def main():
    parser = argparse.ArgumentParser(description='IMPROVED TimeGAN for Bitcoin Dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to Bitcoin CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs_improved', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=24, help='Sequence length')
    parser.add_argument('--n_samples', type=int, default=2000, help='Number of synthetic samples to generate')
    parser.add_argument('--sample_size', type=int, default=500000, help='Sample size for training')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'synthetic'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'processed'), exist_ok=True)
    
    print("=" * 70)
    print("🚀 IMPROVED BITCOIN TIMEGAN PIPELINE")
    print("=" * 70)
    
    try:
        # Step 1: Preprocess data
        print("\n📊 STEP 1: Preprocessing data...")
        train_data, val_data, test_data, scaler, feature_names = preprocess_bitcoin_data_improved(
            args.input, 
            os.path.join(args.output_dir, 'processed'),
            seq_len=args.seq_len,
            sample_size=args.sample_size
        )
        
        # Step 2: Initialize and train TimeGAN
        print("\n🧠 STEP 2: Building and training IMPROVED TimeGAN...")
        trainer = ImprovedTimeGANTrainer(
            seq_len=args.seq_len,
            feature_dim=train_data.shape[2],
            hidden_dim=128,
            z_dim=64
        )
        
        history = trainer.train(
            train_data, val_data, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        # Step 3: Generate synthetic data
        print("\n🎨 STEP 3: Generating synthetic data...")
        synthetic_data = trainer.generate_synthetic_data(args.n_samples, scaler)
        
        # Step 4: Evaluate synthetic data
        print("\n📈 STEP 4: Evaluating synthetic data...")
        real_test_denorm = scaler.inverse_transform(
            test_data.reshape(-1, test_data.shape[-1])
        ).reshape(test_data.shape)
        
        metrics = evaluate_synthetic_data_improved(real_test_denorm, synthetic_data)
        
        # Step 5: Save results
        print("\n💾 STEP 5: Saving results...")
        
        # Save synthetic data
        synthetic_path = os.path.join(args.output_dir, 'synthetic', 'synthetic_bitcoin.npy')
        np.save(synthetic_path, synthetic_data)
        
        # Save models
        trainer.embedder.save_weights(os.path.join(args.output_dir, 'models', 'embedder.weights.h5'))
        trainer.recovery.save_weights(os.path.join(args.output_dir, 'models', 'recovery.weights.h5'))
        trainer.generator.save_weights(os.path.join(args.output_dir, 'models', 'generator.weights.h5'))
        trainer.supervisor.save_weights(os.path.join(args.output_dir, 'models', 'supervisor.weights.h5'))
        trainer.discriminator.save_weights(os.path.join(args.output_dir, 'models', 'discriminator.weights.h5'))
        
        # Save metrics and history
        metrics_path = os.path.join(args.output_dir, 'synthetic', 'evaluation_metrics.json')
        metrics_serializable = {}
        for key, value in metrics.items():
            metrics_serializable[key] = float(value)
        
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics_serializable, f, indent=2)
        
        history_path = os.path.join(args.output_dir, 'synthetic', 'training_history.json')
        history_serializable = {}
        for key, values in history.items():
            history_serializable[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        # Step 6: Create comprehensive analysis
        print("\n📊 STEP 6: Creating comprehensive analysis...")
        create_comprehensive_analysis(
            test_data[:1000], synthetic_data[:1000], history, feature_names, scaler, args.output_dir
        )
        
        # Final report
        print("\n" + "=" * 70)
        print("🎉 IMPROVED TIMEGAN PIPELINE COMPLETED!")
        print("=" * 70)
        print(f"📊 Real data shape: {real_test_denorm.shape}")
        print(f"🎨 Synthetic data shape: {synthetic_data.shape}")
        
        # Quality assessment
        if metrics['mean_difference'] < 1000 and metrics['correlation_difference'] < 0.1:
            print("🎉 EXCELLENT: High-quality synthetic data generated!")
        elif metrics['mean_difference'] < 5000 and metrics['correlation_difference'] < 0.2:
            print("✅ GOOD: Synthetic data generated with acceptable quality")
        else:
            print("⚠️ MODERATE: Synthetic data generated but needs improvement")
            print("   Consider: More training epochs, hyperparameter tuning, or different architecture")
        
        print(f"📈 Quality metrics: {metrics_path}")
        print(f"📊 Training history: {history_path}")
        print(f"💾 Synthetic data: {synthetic_path}")
        print(f"💾 Models: {os.path.join(args.output_dir, 'models')}")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()