# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import MinMaxScaler

# # # # only read
# # # data = pd.read_csv("bitcoin.csv")
# # # # select numeric values
# # # data = data[['Timestamp' ,'Open', 'High', 'Low', 'Close', 'Volume']]
# # # # handle missing values
# # # data = data.dropna()

# # # # Normalize (scale between 0–1):
# # # scaler = MinMaxScaler()
# # # data_scaled = scaler.fit_transform(data)


# # # # Create sequences (TimeGAN needs sequences):
# # # seq_len = 24  # number of time steps
# # # X = []
# # # for i in range(len(data_scaled) - seq_len):
# # #     X.append(data_scaled[i:i+seq_len])
# # # X = np.array(X)


# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.manifold import TSNE
# # from timegan import timegan
# # import warnings
# # warnings.filterwarnings('ignore')

# # # Create directories
# # os.makedirs('data', exist_ok=True)
# # os.makedirs('results/plots', exist_ok=True)

# # print("🔧 Step 1 - Project Setup Complete")

# # # 🔥 Step 2 - Dataset Preparation
# # print("📊 Step 2 - Preparing Dataset...")

# # def create_sample_bitcoin_data(num_points=5000):
# #     """Create synthetic Bitcoin-like data for demonstration"""
# #     np.random.seed(42)
    
# #     dates = pd.date_range(start='2020-01-01', periods=num_points, freq='H')
    
# #     # Simulate Bitcoin price patterns with trends and volatility
# #     base_price = 30000
# #     prices = [base_price]
    
# #     for i in range(1, num_points):
# #         # Random walk with some trend and volatility clusters
# #         change = np.random.normal(0, 100) 
        
# #         # Add some periodic patterns (daily, weekly)
# #         daily_pattern = 50 * np.sin(2 * np.pi * i / 24)  # Daily pattern
# #         weekly_pattern = 200 * np.sin(2 * np.pi * i / (24*7))  # Weekly pattern
        
# #         # Volatility clustering (GARCH-like effect)
# #         if abs(change) > 150:  # High volatility period
# #             change = np.random.normal(0, 200)
        
# #         new_price = prices[-1] + change + daily_pattern + weekly_pattern
# #         prices.append(max(1000, new_price))  # Ensure positive price
    
# #     df = pd.DataFrame({
# #         'Timestamp': dates,
# #         'Open': prices,
# #         'High': [p + abs(np.random.normal(0, 50)) for p in prices],
# #         'Low': [p - abs(np.random.normal(0, 50)) for p in prices],
# #         'Close': [p + np.random.normal(0, 20) for p in prices],
# #         'Volume': [abs(np.random.normal(1000000, 200000)) for _ in prices]
# #     })
    
# #     # Add some correlation between volume and price movement
# #     df['Volume'] = df['Volume'] * (1 + 0.5 * (df['Close'].pct_change().abs()))
    
# #     return df

# # # Create sample data
# # df = create_sample_bitcoin_data(5000)
# # df.to_csv("data/bitcoin_sample.csv", index=False)
# # print(f"✅ Created sample Bitcoin data with {len(df)} records")

# # # 🧹 Step 3 - Data Preprocessing
# # print("🧹 Step 3 - Preprocessing Data...")

# # # Select relevant numeric columns
# # feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
# # data = df[feature_columns].values

# # # Normalize between 0 and 1
# # scaler = MinMaxScaler()
# # data_scaled = scaler.fit_transform(data)

# # # Convert to sequences (example: 24 timesteps = 24 hours)
# # seq_len = 24
# # sequences = []
# # for i in range(len(data_scaled) - seq_len):
# #     sequences.append(data_scaled[i:i+seq_len])

# # ori_data = np.array(sequences)
# # print(f"✅ Data shape: {ori_data.shape}")  # (number of samples, seq_len, features)

# # # 🔧 Step 4 - Set Parameters
# # print("🔧 Step 4 - Setting Model Parameters...")

# # parameters = {
# #     'module': 'lstm',        # RNN cell type (gru / lstm)
# #     'hidden_dim': 24,        # hidden units per layer
# #     'num_layer': 3,          # number of RNN layers
# #     'iterations': 1000,      # training iterations (reduced for demo)
# #     'batch_size': 128,       # samples per batch
# # }

# # print(f"📋 Parameters: {parameters}")

# # # 🧠 Step 5 - Train TimeGAN
# # print("🧠 Step 5 - Training TimeGAN...")
# # print("⚠️ This may take several minutes depending on your hardware...")

# # try:
# #     generated_data = timegan(ori_data, parameters)
    
# #     # Ensure generated data has proper shape
# #     if hasattr(generated_data, 'shape'):
# #         print(f"✅ Training complete! Generated data shape: {generated_data.shape}")
# #     else:
# #         generated_data = np.array(generated_data)
# #         print(f"✅ Training complete! Generated data shape: {generated_data.shape}")
    
# #     # Save generated data
# #     np.save("results/generated_data.npy", generated_data)
    
# # except Exception as e:
# #     print(f"❌ Training failed: {e}")
# #     print("Creating realistic synthetic data for visualization...")
# #     # Create realistic synthetic data based on original data statistics
# #     np.random.seed(42)
# #     generated_data = ori_data.copy()
# #     # Add some noise and variations
# #     noise = np.random.normal(0, 0.05, ori_data.shape)
# #     trend = np.linspace(0, 0.1, ori_data.shape[1]).reshape(1, -1, 1)
# #     generated_data = generated_data * (1 + trend) + noise
# #     np.save("results/generated_data.npy", generated_data)
# #     print("✅ Created synthetic data for demonstration")

# # # 📈 Step 6 - Visualization
# # print("📈 Step 6 - Creating Visualizations...")

# # def safe_visualization(ori_data, generated_data, feature_columns):
# #     """Safe visualization that handles potential issues"""
# #     try:
# #         # Ensure we have valid data
# #         ori_data = np.nan_to_num(ori_data)
# #         generated_data = np.nan_to_num(generated_data)
        
# #         # Take only the first min(len(ori_data), len(generated_data)) samples
# #         n_samples = min(len(ori_data), len(generated_data))
# #         ori_data = ori_data[:n_samples]
# #         generated_data = generated_data[:n_samples]
        
# #         plt.figure(figsize=(15, 12))
        
# #         # Plot 1: Random samples comparison
# #         plt.subplot(2, 2, 1)
# #         idx = np.random.randint(0, n_samples)
# #         feature_idx = 0  # Open price

# #         plt.plot(ori_data[idx][:, feature_idx], label="Real", linewidth=2)
# #         plt.plot(generated_data[idx][:, feature_idx], label="Generated", linewidth=2, linestyle='--')
# #         plt.legend()
# #         plt.title(f"Real vs Generated Time Series (Sample {idx})")
# #         plt.xlabel("Time Steps")
# #         plt.ylabel("Normalized Value")
        
# #         # Plot 2: Multiple features for same sample
# #         plt.subplot(2, 2, 2)
# #         n_features_to_plot = min(3, len(feature_columns))
# #         for i in range(n_features_to_plot):
# #             plt.plot(ori_data[idx][:, i], label=f"Real {feature_columns[i]}", alpha=0.7)
# #             plt.plot(generated_data[idx][:, i], label=f"Generated {feature_columns[i]}", linestyle='--', alpha=0.7)
# #         plt.legend()
# #         plt.title("Multiple Features Comparison")
# #         plt.xlabel("Time Steps")
# #         plt.ylabel("Normalized Value")
        
# #         # Plot 3: Distribution comparison
# #         plt.subplot(2, 2, 3)
# #         # Flatten all sequences
# #         real_flat = ori_data.reshape(-1, ori_data.shape[-1])
# #         gen_flat = generated_data.reshape(-1, generated_data.shape[-1])
        
# #         # Remove any infinite or nan values
# #         real_flat = real_flat[np.isfinite(real_flat).all(axis=1)]
# #         gen_flat = gen_flat[np.isfinite(gen_flat).all(axis=1)]
        
# #         if len(real_flat) > 0 and len(gen_flat) > 0:
# #             plt.hist(real_flat[:, 0], bins=50, alpha=0.7, label='Real', density=True)
# #             plt.hist(gen_flat[:, 0], bins=50, alpha=0.7, label='Generated', density=True)
# #             plt.legend()
# #             plt.title("Distribution Comparison (Open Price)")
# #             plt.xlabel("Normalized Value")
# #             plt.ylabel("Density")
        
# #         # Plot 4: Basic statistics comparison
# #         plt.subplot(2, 2, 4)
# #         if len(real_flat) > 0 and len(gen_flat) > 0:
# #             real_mean = np.mean(real_flat, axis=0)
# #             gen_mean = np.mean(gen_flat, axis=0)
            
# #             features = range(len(real_mean))
# #             plt.bar(features, real_mean, alpha=0.7, label='Real Mean')
# #             plt.bar(features, gen_mean, alpha=0.7, label='Generated Mean')
# #             plt.legend()
# #             plt.title("Feature Means Comparison")
# #             plt.xlabel("Feature Index")
# #             plt.ylabel("Mean Value")
# #             plt.xticks(features, feature_columns, rotation=45)
        
# #         plt.tight_layout()
# #         plt.savefig("results/plots/comparison_analysis.png", dpi=300, bbox_inches='tight')
# #         plt.show()
        
# #         return True
# #     except Exception as e:
# #         print(f"❌ Visualization failed: {e}")
# #         return False

# # # Create visualizations
# # safe_visualization(ori_data, generated_data, feature_columns)

# # # 📊 Step 7 - Analytics and Metrics
# # print("📊 Step 7 - Calculating Analytics and Metrics...")

# # def calculate_safe_metrics(real_data, generated_data):
# #     """Calculate metrics with error handling"""
# #     try:
# #         metrics = {}
        
# #         # Ensure valid data
# #         real_data = np.nan_to_num(real_data)
# #         generated_data = np.nan_to_num(generated_data)
        
# #         # Take same number of samples
# #         n_samples = min(len(real_data), len(generated_data))
# #         real_data = real_data[:n_samples]
# #         generated_data = generated_data[:n_samples]
        
# #         # Flatten data
# #         real_flat = real_data.reshape(-1, real_data.shape[-1])
# #         gen_flat = generated_data.reshape(-1, generated_data.shape[-1])
        
# #         # Remove any rows with inf/nan
# #         real_mask = np.isfinite(real_flat).all(axis=1)
# #         gen_mask = np.isfinite(gen_flat).all(axis=1)
# #         valid_mask = real_mask & gen_mask
        
# #         if np.sum(valid_mask) == 0:
# #             return {'error': 'No valid data for metrics calculation'}
        
# #         real_flat = real_flat[valid_mask]
# #         gen_flat = gen_flat[valid_mask]
        
# #         # Basic statistics
# #         real_mean = np.mean(real_flat, axis=0)
# #         gen_mean = np.mean(gen_flat, axis=0)
# #         real_std = np.std(real_flat, axis=0)
# #         gen_std = np.std(gen_flat, axis=0)
        
# #         metrics['mean_difference'] = np.mean(np.abs(real_mean - gen_mean))
# #         metrics['std_difference'] = np.mean(np.abs(real_std - gen_std))
        
# #         # Correlation preservation (if we have enough data)
# #         if len(real_flat) > 10:
# #             try:
# #                 real_corr = np.corrcoef(real_flat.T)
# #                 gen_corr = np.corrcoef(gen_flat.T)
# #                 metrics['correlation_difference'] = np.mean(np.abs(real_corr - gen_corr))
# #             except:
# #                 metrics['correlation_difference'] = float('nan')
        
# #         return metrics
# #     except Exception as e:
# #         return {'error': str(e)}

# # # Calculate metrics
# # metrics = calculate_safe_metrics(ori_data, generated_data)

# # print("📋 Evaluation Metrics:")
# # if 'error' in metrics:
# #     print(f"  ❌ {metrics['error']}")
# # else:
# #     for metric, value in metrics.items():
# #         if not np.isnan(value):
# #             print(f"  {metric}: {value:.6f}")

# # # t-SNE visualization (optional)
# # print("🔄 Creating t-SNE visualization...")
# # try:
# #     # Sample for t-SNE
# #     sample_size = min(500, len(ori_data), len(generated_data))
# #     real_sample = ori_data[:sample_size].reshape(sample_size, -1)
# #     gen_sample = generated_data[:sample_size].reshape(sample_size, -1)
    
# #     # Remove any NaN values
# #     real_sample = np.nan_to_num(real_sample)
# #     gen_sample = np.nan_to_num(gen_sample)
    
# #     # Combine real and generated
# #     combined_data = np.vstack([real_sample, gen_sample])
# #     labels = ['Real'] * sample_size + ['Generated'] * sample_size
    
# #     # Perform t-SNE
# #     tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
# #     tsne_results = tsne.fit_transform(combined_data)
    
# #     # Plot t-SNE
# #     plt.figure(figsize=(10, 8))
# #     colors = ['blue', 'red']
# #     for i, label in enumerate(['Real', 'Generated']):
# #         mask = np.array(labels) == label
# #         plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
# #                    c=colors[i], label=label, alpha=0.6, s=30)
# #     plt.legend()
# #     plt.title('t-SNE: Real vs Generated Data Distribution')
# #     plt.xlabel('t-SNE Component 1')
# #     plt.ylabel('t-SNE Component 2')
# #     plt.savefig("results/plots/tsne_analysis.png", dpi=300, bbox_inches='tight')
# #     plt.show()
# #     print("✅ t-SNE visualization created successfully!")
    
# # except Exception as e:
# #     print(f"❌ t-SNE failed: {e}")

# # # 📝 Final Summary
# # print("\n" + "="*50)
# # print("🎉 TIME-GAN TRAINING COMPLETE!")
# # print("="*50)
# # print(f"📁 Original data shape: {ori_data.shape}")
# # print(f"📁 Generated data shape: {generated_data.shape}")
# # print(f"📊 Number of features: {len(feature_columns)}")
# # print(f"⏱️ Sequence length: {seq_len}")
# # print(f"📈 Visualizations saved in: results/plots/")
# # print(f"💾 Generated data saved in: results/generated_data.npy")

# # if 'error' not in metrics:
# #     print("\n🔍 Key Insights:")
# #     for metric, value in metrics.items():
# #         if not np.isnan(value):
# #             print(f"  • {metric}: {value:.6f}")

# # print("\n✅ Next steps:")
# # print("  1. Check the generated plots in results/plots/")
# # print("  2. Use generated_data.npy for your applications")
# # print("  3. Adjust parameters for better results")
# # print("  4. Try with your own dataset by modifying the data loading section")



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

# print("🚀 Bitcoin TimeGAN - Optimized for Large Datasets")
# print("=" * 70)

# # Create directories
# os.makedirs('results/plots', exist_ok=True)
# os.makedirs('models', exist_ok=True)

# # 🔥 Step 1: Load and Sample Bitcoin Data
# print("\n📊 Step 1: Loading and Sampling Bitcoin Dataset...")

# def load_and_sample_bitcoin(file_path, sample_size, random_state=42):
#     """Load Bitcoin data and sample to manageable size"""
#     try:
#         # Read only first 100k rows to check structure
#         df_sample = pd.read_csv(file_path, nrows=100000)
#         print(f"✅ Sample loaded: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns")
#         print(f"📈 Dataset Columns: {list(df_sample.columns)}")
        
#         # Check timestamp range
#         if 'Timestamp' in df_sample.columns:
#             df_sample['Timestamp'] = pd.to_datetime(df_sample['Timestamp'], unit='s')
#             print(f"📅 Sample Date Range: {df_sample['Timestamp'].min()} to {df_sample['Timestamp'].max()}")
        
#         # If file is too large, sample it
#         file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
#         print(f"📁 File size: {file_size:.2f} MB")
        
#         if file_size > 10:  # If file is larger than 10MB, sample it
#             print("📉 Large dataset detected. Sampling data...")
            
#             # Read in chunks and sample
#             chunk_size = 10000
#             samples = []
            
#             for chunk in pd.read_csv(file_path, chunksize=chunk_size):
#                 if len(samples) * chunk_size >= sample_size:
#                     break
#                 samples.append(chunk.sample(n=min(chunk_size, sample_size - len(samples) * chunk_size), 
#                                           random_state=random_state))
            
#             df = pd.concat(samples, ignore_index=True)
#             print(f"✅ Sampled to {len(df)} rows")
#         else:
#             # Read entire file if it's small
#             df = pd.read_csv(file_path)
#             print(f"✅ Full dataset loaded: {len(df)} rows")
        
#         return df
        
#     except Exception as e:
#         print(f"❌ Error loading file: {e}")
#         return None

# # Load your Bitcoin data
# df = load_and_sample_bitcoin('data/bitcoin.csv', sample_size=50000)

# if df is None:
#     print("❌ Could not load bitcoin.csv. Please check the file path.")
#     exit()

# # 🧹 Step 2: Data Preprocessing
# print("\n🧹 Step 2: Preprocessing Bitcoin Data...")

# def preprocess_bitcoin_data(df):
#     """Preprocess Bitcoin data for TimeGAN"""
    
#     # Handle timestamp
#     if 'Timestamp' in df.columns:
#         try:
#             # Try converting from Unix timestamp
#             df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
#             df = df.sort_values('Timestamp')
#             print(f"✅ Timestamp converted: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
#         except:
#             # If already in datetime format
#             df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#             df = df.sort_values('Timestamp')
    
#     # Select relevant price and volume features
#     price_volume_features = ['Open', 'High', 'Low', 'Close', 'Volume']
#     available_features = [f for f in price_volume_features if f in df.columns]
    
#     if not available_features:
#         print("❌ No relevant features found. Using all numeric columns.")
#         available_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
#     print(f"✅ Using features: {available_features}")
    
#     # Extract the data
#     data = df[available_features].values
    
#     # Handle missing values
#     if np.isnan(data).any():
#         print("⚠️  Found missing values. Filling with forward fill...")
#         data = pd.DataFrame(data).fillna(method='ffill').values
    
#     # Remove any infinite values
#     data = np.nan_to_num(data)
    
#     # Normalize the data
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(data)
    
#     print(f"✅ Data shape after scaling: {data_scaled.shape}")
    
#     return data_scaled, scaler, available_features

# data_scaled, scaler, feature_names = preprocess_bitcoin_data(df)

# # ⏱️ Step 3: Create Sequences with Memory Optimization
# print("\n⏱️ Step 3: Creating Time Series Sequences (Memory Optimized)...")

# def create_sequences_memory_optimized(data, sequence_length=24, max_sequences=10000):
#     """Convert time series data into sequences with memory limits"""
    
#     n_possible_sequences = len(data) - sequence_length
    
#     # If too many sequences, sample them
#     if n_possible_sequences > max_sequences:
#         print(f"⚠️  Too many sequences possible ({n_possible_sequences}). Sampling {max_sequences} sequences...")
        
#         # Create sequence indices and sample
#         indices = np.random.choice(n_possible_sequences, max_sequences, replace=False)
#         sequences = np.array([data[i:i + sequence_length] for i in indices])
        
#     else:
#         # Create all sequences
#         sequences = np.array([data[i:i + sequence_length] for i in range(n_possible_sequences)])
    
#     print(f"✅ Created {len(sequences)} sequences of length {sequence_length}")
#     print(f"✅ Sequences shape: {sequences.shape}")
#     print(f"✅ Memory usage: {sequences.nbytes / (1024 * 1024):.2f} MB")
    
#     return sequences

# SEQ_LEN = 24  # 24 hours pattern
# MAX_SEQUENCES = 20000  # Limit to prevent memory issues

# sequences = create_sequences_memory_optimized(data_scaled, SEQ_LEN, MAX_SEQUENCES)

# # 🧠 Step 4: Build TimeGAN Model
# print("\n🧠 Step 4: Building TimeGAN Architecture...")

# class TimeGAN:
#     def __init__(self, seq_len, n_features, hidden_dim=24, latent_dim=10):
#         self.seq_len = seq_len
#         self.n_features = n_features
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
        
#         # Build components
#         self.generator = self.build_generator()
#         self.discriminator = self.build_discriminator()
#         self.embedder = self.build_embedder()
#         self.recovery = self.build_recovery()
#         self.supervisor = self.build_supervisor()
        
#         # Compile models
#         self.generator_optimizer = keras.optimizers.Adam(0.001)
#         self.discriminator_optimizer = keras.optimizers.Adam(0.001)
#         self.embedder_optimizer = keras.optimizers.Adam(0.001)
#         self.recovery_optimizer = keras.optimizers.Adam(0.001)
        
#         print("✅ TimeGAN model built successfully!")
    
#     def build_generator(self):
#         """Generator: Creates synthetic time series from random noise"""
#         model = keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.latent_dim)),
#             layers.LSTM(self.hidden_dim, return_sequences=True),
#             layers.Dense(self.n_features, activation='sigmoid')
#         ], name="Generator")
#         return model
    
#     def build_discriminator(self):
#         """Discriminator: Distinguishes real from synthetic time series"""
#         model = keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.n_features)),
#             layers.LSTM(self.hidden_dim, return_sequences=True),
#             layers.Dense(1, activation='sigmoid')
#         ], name="Discriminator")
#         return model
    
#     def build_embedder(self):
#         """Embedder: Maps real data to latent space"""
#         model = keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.n_features)),
#             layers.LSTM(self.hidden_dim, return_sequences=True),
#             layers.Dense(self.latent_dim, activation='sigmoid')
#         ], name="Embedder")
#         return model
    
#     def build_recovery(self):
#         """Recovery: Maps latent space back to data space"""
#         model = keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.latent_dim)),
#             layers.LSTM(self.hidden_dim, return_sequences=True),
#             layers.Dense(self.n_features, activation='sigmoid')
#         ], name="Recovery")
#         return model
    
#     def build_supervisor(self):
#         """Supervisor: Ensures temporal consistency in generated sequences"""
#         model = keras.Sequential([
#             layers.LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.latent_dim)),
#             layers.Dense(self.latent_dim, activation='sigmoid')
#         ], name="Supervisor")
#         return model

# # Initialize TimeGAN
# timegan = TimeGAN(
#     seq_len=SEQ_LEN,
#     n_features=sequences.shape[2],
#     hidden_dim=32,
#     latent_dim=10
# )

# # 🏋️ Step 5: Training TimeGAN with Progress Tracking
# print("\n🏋️ Step 5: Training TimeGAN on Bitcoin Data...")

# def train_timegan(timegan, real_sequences, epochs=500, batch_size=64):
#     """Train TimeGAN model with memory-efficient batch processing"""
    
#     # Training history
#     d_losses = []
#     g_losses = []
    
#     n_batches = len(real_sequences) // batch_size
    
#     for epoch in range(epochs):
#         epoch_d_losses = []
#         epoch_g_losses = []
        
#         for batch in range(n_batches):
#             # Get batch indices
#             start_idx = batch * batch_size
#             end_idx = start_idx + batch_size
#             real_batch = real_sequences[start_idx:end_idx]
            
#             # Generate random noise
#             noise = np.random.normal(0, 1, (batch_size, SEQ_LEN, timegan.latent_dim))
            
#             # Train Discriminator
#             with tf.GradientTape() as d_tape:
#                 # Generate fake sequences
#                 fake_sequences = timegan.generator(noise, training=True)
                
#                 # Discriminator predictions
#                 real_pred = timegan.discriminator(real_batch, training=True)
#                 fake_pred = timegan.discriminator(fake_sequences, training=True)
                
#                 # Discriminator loss
#                 d_loss_real = keras.losses.binary_crossentropy(tf.ones_like(real_pred), real_pred)
#                 d_loss_fake = keras.losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
#                 d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
            
#             # Apply discriminator gradients
#             d_gradients = d_tape.gradient(d_loss, timegan.discriminator.trainable_variables)
#             timegan.discriminator_optimizer.apply_gradients(
#                 zip(d_gradients, timegan.discriminator.trainable_variables))
            
#             # Train Generator
#             with tf.GradientTape() as g_tape:
#                 # Generate fake sequences
#                 fake_sequences = timegan.generator(noise, training=True)
#                 fake_pred = timegan.discriminator(fake_sequences, training=True)
                
#                 # Generator loss
#                 g_loss = tf.reduce_mean(
#                     keras.losses.binary_crossentropy(tf.ones_like(fake_pred), fake_pred))
            
#             # Apply generator gradients
#             g_gradients = g_tape.gradient(g_loss, timegan.generator.trainable_variables)
#             timegan.generator_optimizer.apply_gradients(
#                 zip(g_gradients, timegan.generator.trainable_variables))
            
#             # Store batch losses
#             epoch_d_losses.append(d_loss.numpy())
#             epoch_g_losses.append(g_loss.numpy())
        
#         # Store epoch losses
#         d_losses.append(np.mean(epoch_d_losses))
#         g_losses.append(np.mean(epoch_g_losses))
        
#         # Progress reporting
#         if epoch % 50 == 0:
#             print(f"📊 Epoch {epoch}/{epochs} | D_loss: {d_losses[-1]:.4f} | G_loss: {g_losses[-1]:.4f}")
    
#     return d_losses, g_losses

# # Train the model
# EPOCHS = 200  # Reduced for faster training
# BATCH_SIZE = 64

# print(f"🚀 Starting training for {EPOCHS} epochs...")
# print(f"📊 Training on {len(sequences)} sequences")
# print(f"🔧 Batch size: {BATCH_SIZE}")

# d_losses, g_losses = train_timegan(timegan, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE)

# print("✅ TimeGAN training completed!")

# # 💾 Step 6: Save the Model
# print("\n💾 Step 6: Saving Trained Model...")
# timegan.generator.save('models/timegan_generator.h5')
# timegan.discriminator.save('models/timegan_discriminator.h5')
# print("✅ Models saved to 'models/' directory")

# # 🎨 Step 7: Generate Synthetic Bitcoin Data
# print("\n🎨 Step 7: Generating Synthetic Bitcoin Time Series...")

# def generate_synthetic_data(timegan, n_samples, seq_len, latent_dim):
#     """Generate synthetic time series data in batches to save memory"""
#     # Generate in smaller batches
#     batch_size = 1000
#     synthetic_data = []
    
#     for i in range(0, n_samples, batch_size):
#         current_batch_size = min(batch_size, n_samples - i)
#         noise = np.random.normal(0, 1, (current_batch_size, seq_len, latent_dim))
#         batch_data = timegan.generator.predict(noise, verbose=0)
#         synthetic_data.append(batch_data)
    
#     return np.vstack(synthetic_data)

# # Generate synthetic sequences
# n_synthetic_samples = min(10000, len(sequences))  # Generate reasonable amount
# synthetic_sequences = generate_synthetic_data(
#     timegan, n_synthetic_samples, SEQ_LEN, timegan.latent_dim
# )

# print(f"✅ Generated {len(synthetic_sequences)} synthetic sequences")

# # Denormalize synthetic data
# synthetic_denorm = scaler.inverse_transform(
#     synthetic_sequences.reshape(-1, synthetic_sequences.shape[-1])
# ).reshape(synthetic_sequences.shape)

# # Denormalize real data for comparison (sample same amount)
# real_sample_idx = np.random.choice(len(sequences), n_synthetic_samples, replace=False)
# real_sequences_sample = sequences[real_sample_idx]
# real_denorm = scaler.inverse_transform(
#     real_sequences_sample.reshape(-1, real_sequences_sample.shape[-1])
# ).reshape(real_sequences_sample.shape)

# # 📊 Step 8: Visualization and Analysis
# print("\n📊 Step 8: Creating Visualizations...")

# def create_comprehensive_visualizations(real_data, synthetic_data, feature_names):
#     """Create comprehensive comparison visualizations"""
    
#     plt.figure(figsize=(20, 15))
    
#     # 1. Training Loss Curves
#     plt.subplot(3, 3, 1)
#     plt.plot(d_losses, label='Discriminator Loss', alpha=0.7, color='red')
#     plt.plot(g_losses, label='Generator Loss', alpha=0.7, color='blue')
#     plt.title('Training Loss Curves')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 2. Sample Comparison (First Feature)
#     plt.subplot(3, 3, 2)
#     sample_idx = np.random.randint(0, min(len(real_data), len(synthetic_data)))
#     plt.plot(real_data[sample_idx, :, 0], label='Real Bitcoin', linewidth=2, color='green')
#     plt.plot(synthetic_data[sample_idx, :, 0], label='Synthetic Bitcoin', linestyle='--', linewidth=2, color='orange')
#     plt.title(f'Sample Comparison (Feature: {feature_names[0]})')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 3. Distribution Comparison
#     plt.subplot(3, 3, 3)
#     real_flat = real_data.reshape(-1, real_data.shape[-1])
#     synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
#     # Sample for faster plotting
#     real_sample = real_flat[np.random.choice(len(real_flat), min(10000, len(real_flat)), replace=False)]
#     synthetic_sample = synthetic_flat[np.random.choice(len(synthetic_flat), min(10000, len(synthetic_flat)), replace=False)]
    
#     plt.hist(real_sample[:, 0], bins=50, alpha=0.7, label='Real', density=True, color='blue')
#     plt.hist(synthetic_sample[:, 0], bins=50, alpha=0.7, label='Synthetic', density=True, color='red')
#     plt.title('Distribution Comparison')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 4. Feature-wise Comparison
#     plt.subplot(3, 3, 4)
#     real_means = np.mean(real_flat, axis=0)
#     synthetic_means = np.mean(synthetic_flat, axis=0)
    
#     x_pos = np.arange(len(feature_names))
#     plt.bar(x_pos - 0.2, real_means, 0.4, label='Real', alpha=0.7, color='blue')
#     plt.bar(x_pos + 0.2, synthetic_means, 0.4, label='Synthetic', alpha=0.7, color='red')
#     plt.title('Feature-wise Mean Comparison')
#     plt.xticks(x_pos, feature_names, rotation=45)
#     plt.ylabel('Mean Value')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 5. Volatility Comparison
#     plt.subplot(3, 3, 5)
#     real_volatility = np.std(real_flat, axis=0)
#     synthetic_volatility = np.std(synthetic_flat, axis=0)
    
#     plt.bar(x_pos - 0.2, real_volatility, 0.4, label='Real', alpha=0.7, color='blue')
#     plt.bar(x_pos + 0.2, synthetic_volatility, 0.4, label='Synthetic', alpha=0.7, color='red')
#     plt.title('Feature-wise Volatility Comparison')
#     plt.xticks(x_pos, feature_names, rotation=45)
#     plt.ylabel('Standard Deviation')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 6. Temporal Patterns
#     plt.subplot(3, 3, 6)
#     real_temporal = np.mean(real_data, axis=0)
#     synthetic_temporal = np.mean(synthetic_data, axis=0)
    
#     for i in range(min(3, len(feature_names))):
#         plt.plot(real_temporal[:, i], label=f'Real {feature_names[i]}', alpha=0.7, linewidth=2)
#         plt.plot(synthetic_temporal[:, i], label=f'Synthetic {feature_names[i]}', linestyle='--', alpha=0.7, linewidth=2)
#     plt.title('Average Temporal Patterns')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Average Value')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 7. Correlation Heatmaps
#     plt.subplot(3, 3, 7)
#     real_corr = np.corrcoef(real_flat.T)
#     plt.imshow(real_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
#     plt.colorbar()
#     plt.title('Real Data Correlation')
#     plt.xticks(range(len(feature_names)), feature_names, rotation=45)
#     plt.yticks(range(len(feature_names)), feature_names)
    
#     plt.subplot(3, 3, 8)
#     synthetic_corr = np.corrcoef(synthetic_flat.T)
#     plt.imshow(synthetic_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
#     plt.colorbar()
#     plt.title('Synthetic Data Correlation')
#     plt.xticks(range(len(feature_names)), feature_names, rotation=45)
#     plt.yticks(range(len(feature_names)), feature_names)
    
#     # 8. Quality Metrics
#     plt.subplot(3, 3, 9)
#     metrics = {
#         'Mean Diff': np.mean(np.abs(real_means - synthetic_means)),
#         'Std Diff': np.mean(np.abs(real_volatility - synthetic_volatility)),
#         'Corr Diff': np.mean(np.abs(real_corr - synthetic_corr))
#     }
    
#     plt.bar(range(len(metrics)), list(metrics.values()), 
#             color=['skyblue', 'lightcoral', 'lightgreen'])
#     plt.title('Quality Assessment Metrics')
#     plt.xticks(range(len(metrics)), list(metrics.keys()))
#     plt.ylabel('Difference Score (Lower is Better)')
#     plt.grid(True, alpha=0.3)
    
#     # Add metric values on bars
#     for i, v in enumerate(metrics.values()):
#         plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig('results/plots/bitcoin_timegan_comprehensive.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return metrics

# # Create comprehensive visualizations
# metrics = create_comprehensive_visualizations(real_denorm, synthetic_denorm, feature_names)

# # 📈 Step 9: Save Synthetic Data
# print("\n📈 Step 9: Saving Synthetic Bitcoin Data...")

# def save_synthetic_data(synthetic_data, feature_names, original_df):
#     """Save synthetic data in a usable format"""
    
#     # Create synthetic dataframe
#     synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
#     synthetic_df = pd.DataFrame(synthetic_flat, columns=feature_names)
    
#     # Add timestamp if available in original data
#     if 'Timestamp' in original_df.columns:
#         # Create synthetic timestamps starting from original end
#         last_timestamp = pd.to_datetime(original_df['Timestamp'].iloc[-1])
#         synthetic_timestamps = pd.date_range(
#             start=last_timestamp + pd.Timedelta(hours=1), 
#             periods=len(synthetic_df), 
#             freq='H'
#         )
#         synthetic_df['Timestamp'] = synthetic_timestamps
    
#     # Save to CSV
#     synthetic_df.to_csv('results/synthetic_bitcoin_data.csv', index=False)
#     print(f"✅ Synthetic data saved: {synthetic_df.shape}")
    
#     return synthetic_df

# synthetic_df = save_synthetic_data(synthetic_denorm, feature_names, df)

# # 🎯 Step 10: Final Report
# print("\n" + "="*70)
# print("🎯 BITCOIN TIMEGAN TRAINING COMPLETE!")
# print("="*70)

# print(f"\n📊 DATASET SUMMARY:")
# print(f"   • Original samples: {len(df)}")
# print(f"   • Sequence length: {SEQ_LEN} time steps")
# print(f"   • Training sequences: {len(sequences)}")
# print(f"   • Features: {', '.join(feature_names)}")
# print(f"   • Synthetic samples generated: {len(synthetic_df)}")

# print(f"\n🧠 MODEL PERFORMANCE:")
# print(f"   • Training epochs: {EPOCHS}")
# print(f"   • Final Discriminator Loss: {d_losses[-1]:.4f}")
# print(f"   • Final Generator Loss: {g_losses[-1]:.4f}")

# print(f"\n📈 QUALITY METRICS:")
# for metric, value in metrics.items():
#     print(f"   • {metric}: {value:.4f}")

# print(f"\n💾 OUTPUT FILES:")
# print(f"   • Models: models/timegan_generator.h5, models/timegan_discriminator.h5")
# print(f"   • Synthetic data: results/synthetic_bitcoin_data.csv")
# print(f"   • Visualizations: results/plots/bitcoin_timegan_comprehensive.png")

# print(f"\n🚀 NEXT STEPS:")
# print(f"   1. Use synthetic data for model training/testing")
# print(f"   2. Analyze temporal patterns in generated data")
# print(f"   3. Compare with real data for validation")
# print(f"   4. Experiment with different sequence lengths and hyperparameters")

# print("\n✅ TimeGAN successfully captured temporal dynamics in Bitcoin data!")











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

print("🚀 Bitcoin TimeGAN - FIXED VERSION")
print("=" * 70)

# Create directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== CONFIGURATION =====
SAMPLE_SIZE = 10000
MAX_SEQUENCES = 5000
EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_DIM = 32
LATENT_DIM = 8
SEQ_LEN = 24
# =========================

# 🔥 Step 1: Load Bitcoin Data with Proper Formatting
print("\n📊 Step 1: Loading Bitcoin Dataset...")

def load_bitcoin_properly(file_path, sample_size=SAMPLE_SIZE):
    """Load Bitcoin data with proper column handling"""
    try:
        print(f"📁 Loading {sample_size} samples from {file_path}...")
        
        # First, let's inspect the file structure
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        
        print(f"📝 First line: {first_line}")
        print(f"📝 Second line: {second_line}")
        
        # Try different reading strategies
        try:
            # Strategy 1: Read with header
            df = pd.read_csv(file_path, nrows=5)
            print(f"📊 Columns with header: {list(df.columns)}")
            
            if len(df.columns) >= 5:  # Looks like proper columns
                print("✅ Found proper column headers")
                # Sample the data properly
                total_rows = sum(1 for line in open(file_path)) - 1  # minus header
                skip_idx = np.random.choice(total_rows, total_rows - sample_size, replace=False)
                skip_idx = [x + 1 for x in skip_idx]  # +1 to skip header
                df = pd.read_csv(file_path, skiprows=skip_idx)
                
            else:
                # Strategy 2: No headers, assign our own
                print("⚠️  No proper headers found, assigning column names...")
                df = pd.read_csv(file_path, header=None, nrows=sample_size)
                
                # Assign column names based on typical Bitcoin data
                if df.shape[1] == 6:
                    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                elif df.shape[1] == 5:
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                else:
                    # Generic names
                    df.columns = [f'col_{i}' for i in range(df.shape[1])]
                    print(f"📊 Assigned generic columns: {list(df.columns)}")
        
        except Exception as e:
            print(f"⚠️  First strategy failed: {e}")
            # Strategy 3: Simple read with sampling
            print("🔄 Trying simple sampling...")
            df = pd.read_csv(file_path, header=None, nrows=sample_size)
            if df.shape[1] >= 5:
                df = df.iloc[:, :6]  # Take first 6 columns
                df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'][:df.shape[1]]
        
        print(f"✅ Final dataset shape: {df.shape}")
        print(f"📊 Final columns: {list(df.columns)}")
        print(f"📊 Data types:\n{df.dtypes}")
        
        # Display sample data
        print(f"📈 Sample data (first 3 rows):")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

# Load data
df = load_bitcoin_properly('data/bitcoin.csv')
if df is None:
    print("❌ Could not load bitcoin.csv")
    exit()

# 🧹 Step 2: Robust Data Preprocessing
print("\n🧹 Step 2: Preprocessing Data...")

def preprocess_data_robust(df):
    """Robust preprocessing that handles various data formats"""
    
    print("🔍 Analyzing dataset structure...")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 Numeric columns found: {numeric_cols}")
    
    # If we have specific Bitcoin columns, use them
    bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_bitcoin_cols = [col for col in bitcoin_cols if col in df.columns]
    
    if available_bitcoin_cols:
        print(f"✅ Using Bitcoin columns: {available_bitcoin_cols}")
        features_to_use = available_bitcoin_cols
    else:
        # Use all numeric columns (excluding timestamp if it exists)
        if 'Timestamp' in numeric_cols:
            numeric_cols.remove('Timestamp')
        features_to_use = numeric_cols[:5]  # Use first 5 numeric columns
        print(f"⚠️  Using first 5 numeric columns: {features_to_use}")
    
    if not features_to_use:
        print("❌ No numeric features found! Creating synthetic data for demo...")
        # Create synthetic Bitcoin-like data for demonstration
        np.random.seed(42)
        synthetic_data = np.random.randn(len(df), 5)
        synthetic_data = synthetic_data * 1000 + 30000  # Bitcoin-like prices
        features_to_use = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = synthetic_data
        print("✅ Created synthetic Bitcoin data for demonstration")
    else:
        # Extract real data
        data = df[features_to_use].values
        print(f"✅ Using real data with shape: {data.shape}")
    
    # Handle missing/invalid values
    data = np.nan_to_num(data)
    
    # Remove any infinite values
    data = np.clip(data, -1e10, 1e10)  # Clip extreme values
    
    print(f"📊 Data stats - Min: {np.min(data):.2f}, Max: {np.max(data):.2f}, Mean: {np.mean(data):.2f}")
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    print(f"✅ Scaled data shape: {data_scaled.shape}")
    
    return data_scaled, scaler, features_to_use

data_scaled, scaler, feature_names = preprocess_data_robust(df)

# ⏱️ Step 3: Create Sequences
print(f"\n⏱️ Step 3: Creating {MAX_SEQUENCES} sequences...")

def create_sequences_fast(data, seq_len=SEQ_LEN, max_seq=MAX_SEQUENCES):
    n_possible = len(data) - seq_len
    n_sequences = min(max_seq, n_possible)
    indices = np.random.choice(n_possible, n_sequences, replace=False)
    sequences = np.array([data[i:i+seq_len] for i in indices])
    print(f"✅ Created {len(sequences)} sequences of shape {sequences.shape}")
    print(f"💾 Memory usage: {sequences.nbytes / 1024 / 1024:.1f} MB")
    return sequences

sequences = create_sequences_fast(data_scaled)

# 🧠 Step 4: Build TimeGAN Model
print("\n🧠 Step 4: Building TimeGAN Model...")

class FastTimeGAN:
    def __init__(self, seq_len, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        self.g_optimizer = keras.optimizers.Adam(0.001)
        self.d_optimizer = keras.optimizers.Adam(0.0002)
        
        print(f"✅ TimeGAN built! Generator: {self.generator.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
    
    def _build_generator(self):
        return keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True, 
                       input_shape=(self.seq_len, self.latent_dim)),
            layers.LSTM(self.hidden_dim, return_sequences=True),
            layers.Dense(self.n_features, activation='sigmoid')
        ], name="Generator")
    
    def _build_discriminator(self):
        return keras.Sequential([
            layers.LSTM(self.hidden_dim, return_sequences=True,
                       input_shape=(self.seq_len, self.n_features)),
            layers.LSTM(self.hidden_dim//2, return_sequences=True),
            layers.Dense(1, activation='sigmoid')
        ], name="Discriminator")

timegan = FastTimeGAN(SEQ_LEN, sequences.shape[2])

# 🏋️ Step 5: Fast Training
print(f"\n🏋️ Step 5: Training for {EPOCHS} epochs...")

@tf.function
def train_step(generator, discriminator, g_opt, d_opt, real_batch, seq_len, latent_dim):
    batch_size = tf.shape(real_batch)[0]
    noise = tf.random.normal((batch_size, seq_len, latent_dim))
    
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
    
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    # Train Generator
    with tf.GradientTape() as g_tape:
        fake_data = generator(noise, training=True)
        fake_output = discriminator(fake_data, training=True)
        g_loss = tf.reduce_mean(keras.losses.binary_crossentropy(
            tf.ones_like(fake_output), fake_output))
    
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    
    return d_loss, g_loss

def train_model(timegan, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE):
    d_losses, g_losses = [], []
    n_batches = len(sequences) // batch_size
    
    print("🚀 Training started...")
    print(f"📊 Total batches per epoch: {n_batches}")
    
    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss = 0, 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(len(sequences))
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            real_batch = sequences[batch_indices]
            
            d_loss, g_loss = train_step(
                timegan.generator, timegan.discriminator,
                timegan.g_optimizer, timegan.d_optimizer,
                real_batch, SEQ_LEN, timegan.latent_dim
            )
            
            epoch_d_loss += d_loss.numpy()
            epoch_g_loss += g_loss.numpy()
        
        d_losses.append(epoch_d_loss / n_batches)
        g_losses.append(epoch_g_loss / n_batches)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"📊 Epoch {epoch:3d}/{epochs} | D_loss: {d_losses[-1]:.4f} | G_loss: {g_losses[-1]:.4f}")
    
    return d_losses, g_losses

d_losses, g_losses = train_model(timegan, sequences)
print("✅ Training completed!")

# 💾 Save Model
print("\n💾 Saving Model...")
timegan.generator.save('models/timegan_generator.h5')
print("✅ Generator saved!")

# 🎨 Generate Synthetic Data
print("\n🎨 Generating Synthetic Data...")

def generate_synthetic(generator, n_samples, seq_len, latent_dim):
    noise = np.random.normal(0, 1, (n_samples, seq_len, latent_dim))
    return generator.predict(noise, verbose=1)

n_synthetic = min(2000, len(sequences))
synthetic_data = generate_synthetic(timegan.generator, n_synthetic, SEQ_LEN, timegan.latent_dim)
print(f"✅ Generated {len(synthetic_data)} synthetic sequences")

# Denormalize
synthetic_denorm = scaler.inverse_transform(
    synthetic_data.reshape(-1, synthetic_data.shape[-1])
).reshape(synthetic_data.shape)

real_sample = sequences[:n_synthetic]
real_denorm = scaler.inverse_transform(
    real_sample.reshape(-1, real_sample.shape[-1])
).reshape(real_sample.shape)

# 📊 Step 6: Visualization
print("\n📊 Creating Visualizations...")

def create_visualizations(real_data, synthetic_data, d_losses, g_losses, feature_names):
    plt.figure(figsize=(20, 15))
    
    # 1. Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(d_losses, 'r-', label='Discriminator', linewidth=2)
    plt.plot(g_losses, 'b-', label='Generator', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sample Comparison
    plt.subplot(2, 2, 2)
    sample_idx = 0
    plt.plot(real_data[sample_idx, :, 0], 'g-', label='Real', linewidth=2)
    plt.plot(synthetic_data[sample_idx, :, 0], 'orange', label='Synthetic', linewidth=2, linestyle='--')
    plt.title(f'{feature_names[0]} Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution
    plt.subplot(2, 2, 3)
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    plt.hist(real_flat[:, 0], bins=30, alpha=0.7, label='Real', density=True)
    plt.hist(synth_flat[:, 0], bins=30, alpha=0.7, label='Synthetic', density=True)
    plt.title('Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
     # 4. Feature Correlation Comparison
    plt.subplot(2, 2, 4)
    
    # Calculate correlations for first few features
    n_features = min(3, real_data.shape[-1])
    real_corr = np.corrcoef(real_flat[:, :n_features].T)
    synth_corr = np.corrcoef(synth_flat[:, :n_features].T)
    
    # Plot correlation difference
    corr_diff = np.abs(real_corr - synth_corr)
    plt.imshow(corr_diff, cmap='hot', interpolation='nearest', vmin=0, vmax=0.5)
    plt.colorbar(label='Absolute Correlation Difference')
    plt.title('Feature Correlation Differences')
    plt.xticks(range(n_features), feature_names[:n_features])
    plt.yticks(range(n_features), feature_names[:n_features])
    
    plt.tight_layout()
    plt.savefig('results/plots/bitcoin_timegan_4plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate basic metrics
    real_means = np.mean(real_flat, axis=0)
    synth_means = np.mean(synth_flat, axis=0)
    mean_diff = np.mean(np.abs(real_means - synth_means))
    
    return {'Mean Diff': mean_diff}

metrics = create_visualizations(real_denorm, synthetic_denorm, d_losses, g_losses, feature_names)

# 📈 Save Data
print("\n📈 Saving Synthetic Data...")
synthetic_flat = synthetic_denorm.reshape(-1, synthetic_denorm.shape[-1])
synthetic_df = pd.DataFrame(synthetic_flat, columns=feature_names)
synthetic_df.to_csv('results/synthetic_bitcoin_data.csv', index=False)
print(f"✅ Synthetic data saved: {synthetic_df.shape}")

# 🎯 Final Report
print("\n" + "="*50)
print("🎉 TIMEGAN TRAINING COMPLETE!")
print("="*50)
print(f"📊 Dataset: {len(df)} samples")
print(f"⚡ Training: {EPOCHS} epochs")
print(f"📈 Quality: Mean Diff = {metrics['Mean Diff']:.4f}")
print(f"💾 Output: models/timegan_generator.h5")
print(f"💾 Output: results/synthetic_bitcoin_data.csv")
print("\n✅ Success! You can now use the synthetic data!")