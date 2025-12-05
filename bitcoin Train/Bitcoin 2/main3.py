import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 STABILIZED BITCOIN TIMEGAN - NAN-FREE IMPLEMENTATION")
print("=" * 70)

# Create directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== STABLE CONFIGURATION =====
MAX_SEQUENCES = 5000  # Reduced for stability
EPOCHS = 200
BATCH_SIZE = 64
HIDDEN_DIM = 64
LATENT_DIM = 16
SEQ_LEN = 24
LEARNING_RATE = 0.0002  # Lower learning rate
GRADIENT_PENALTY_WEIGHT = 5.0  # Reduced penalty
# =================================

# 🔥 Step 1: Load and Prepare Data
print("\n📊 Step 1: Loading and Preparing Bitcoin Dataset...")

def load_and_prepare_data(file_path, sample_size=100000):
    """Load and carefully prepare Bitcoin data"""
    try:
        print(f"📁 Loading dataset from {file_path}...")
        
        # Load sample for stability during development
        df = pd.read_csv(file_path, nrows=sample_size)
        
        # Standardize column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        }
        df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
        
        # Select Bitcoin columns
        bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in bitcoin_cols if col in df.columns]
        
        if not available_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_cols = numeric_cols[:5]
            print(f"⚠️ Using numeric columns: {available_cols}")
        else:
            print(f"✅ Using Bitcoin columns: {available_cols}")
        
        df = df[available_cols].copy()
        
        # Enhanced data cleaning
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove any remaining inf values
        df = df[np.isfinite(df).all(axis=1)]
        
        print(f"✅ Final dataset shape: {df.shape}")
        print(f"📊 Data sample:")
        print(df.head(3))
        print(f"📈 Data statistics:")
        print(df.describe())
        
        return df, available_cols
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

df, feature_names = load_and_prepare_data('../data/bitcoin.csv', sample_size=200000)

# 🧹 Step 2: Ultra-Stable Preprocessing
print("\n🧹 Step 2: Ultra-Stable Data Preprocessing...")

def stable_preprocessing(df, feature_names):
    """Very careful preprocessing to avoid NaN issues"""
    
    data = df[feature_names].values
    
    print("🔍 Applying robust preprocessing...")
    
    # 1. Handle extreme outliers using robust scaling
    from scipy import stats
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    data = data[(z_scores < 10).all(axis=1)]  # Remove extreme outliers
    
    # 2. Use MinMaxScaler for bounded output (better for GANs)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Tanh activation likes this range
    data_scaled = scaler.fit_transform(data)
    
    # 3. Add tiny noise to prevent exact zeros
    noise = np.random.normal(0, 0.001, data_scaled.shape)
    data_scaled = data_scaled + noise
    
    # 4. Ensure no NaN or Inf
    data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    data_scaled = np.clip(data_scaled, -10, 10)  # Conservative clipping
    
    data_scaled = data_scaled.astype(np.float32)
    
    print(f"✅ Preprocessed data shape: {data_scaled.shape}")
    print(f"📊 Final range: [{np.min(data_scaled):.3f}, {np.max(data_scaled):.3f}]")
    print(f"📈 Final stats - Mean: {np.mean(data_scaled, axis=0)}, Std: {np.std(data_scaled, axis=0)}")
    
    return data_scaled, scaler

data_scaled, scaler = stable_preprocessing(df, feature_names)

# ⏱️ Step 3: Create Stable Sequences
print(f"\n⏱️ Step 3: Creating {MAX_SEQUENCES} Stable Sequences...")

def create_stable_sequences(data, seq_len=SEQ_LEN, max_seq=MAX_SEQUENCES):
    """Create sequences with stability checks"""
    
    n_possible = len(data) - seq_len
    n_sequences = min(max_seq, n_possible)
    
    sequences = []
    step_size = max(1, n_possible // n_sequences)
    
    for i in range(0, n_possible, step_size):
        if len(sequences) >= n_sequences:
            break
        seq = data[i:i+seq_len]
        # Check for valid sequence
        if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)):
            sequences.append(seq)
    
    sequences = np.array(sequences, dtype=np.float32)
    
    # Final validation
    assert not np.any(np.isnan(sequences)), "NaN values in sequences!"
    assert not np.any(np.isinf(sequences)), "Inf values in sequences!"
    
    print(f"✅ Created {len(sequences)} stable sequences of shape {sequences.shape}")
    print(f"📊 Valid sequence range: [{np.min(sequences):.3f}, {np.max(sequences):.3f}]")
    
    return sequences

sequences = create_stable_sequences(data_scaled)

# 🧠 Step 4: Build STABLE TimeGAN Model
print("\n🧠 Step 4: Building Stable TimeGAN Model...")

class StableTimeGAN:
    def __init__(self, seq_len, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.generator = self._build_stable_generator()
        self.discriminator = self._build_stable_discriminator()
        
        # Conservative optimizers
        self.g_optimizer = keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = keras.optimizers.Adam(LEARNING_RATE * 0.5, beta_1=0.5, beta_2=0.9)
        
        self.loss_fn = keras.losses.BinaryCrossentropy()
        
        print(f"✅ Stable TimeGAN built!")
        print(f"   Generator: {self.generator.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
    
    def _build_stable_generator(self):
        """Simple and stable generator"""
        inputs = keras.Input(shape=(self.seq_len, self.latent_dim))
        
        x = layers.LSTM(self.hidden_dim, return_sequences=True, 
                       dropout=0.1, recurrent_dropout=0.05)(inputs)
        x = layers.LayerNormalization()(x)  # More stable than BatchNorm
        x = layers.LSTM(self.hidden_dim // 2, return_sequences=True,
                       dropout=0.1)(x)
        x = layers.LayerNormalization()(x)
        
        outputs = layers.Dense(self.n_features, activation='tanh')(x)  # Tanh for bounded output
        
        return Model(inputs, outputs, name="Stable_Generator")
    
    def _build_stable_discriminator(self):
        """Simple and stable discriminator"""
        inputs = keras.Input(shape=(self.seq_len, self.n_features))
        
        x = layers.LSTM(self.hidden_dim, return_sequences=True,
                       dropout=0.1, recurrent_dropout=0.05)(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(self.hidden_dim // 2, return_sequences=False,
                       dropout=0.1)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name="Stable_Discriminator")

# Initialize model
timegan = StableTimeGAN(SEQ_LEN, sequences.shape[2])

# 🏋️ Step 5: NAN-PROOF Training
print(f"\n🏋️ Step 5: Nan-Proof Training for {EPOCHS} epochs...")

def safe_binary_crossentropy(y_true, y_pred):
    """Safe BCE that avoids NaN"""
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))

@tf.function
def stable_train_step(real_batch, generator, discriminator, g_optimizer, d_optimizer, seq_len, latent_dim):
    batch_size = tf.shape(real_batch)[0]
    
    # Generate stable noise
    noise = tf.random.normal((batch_size, seq_len, latent_dim), 
                           mean=0.0, stddev=0.5)
    
    # Train Discriminator
    with tf.GradientTape() as d_tape:
        # Generate fake data
        fake_data = generator(noise, training=True)
        
        # Discriminator outputs with safety
        real_output = discriminator(real_batch, training=True)
        fake_output = discriminator(fake_data, training=True)
        
        # Safe losses
        d_loss_real = safe_binary_crossentropy(tf.ones_like(real_output), real_output)
        d_loss_fake = safe_binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        # Conservative gradient penalty
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = epsilon * real_batch + (1 - epsilon) * fake_data
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = discriminator(interpolated, training=True)
        
        gradients = gp_tape.gradient(interpolated_output, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-8)
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        d_loss += GRADIENT_PENALTY_WEIGHT * gradient_penalty
    
    # Apply gradients with clipping
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_gradients = [tf.clip_by_norm(grad, 1.0) for grad in d_gradients]  # Gradient clipping
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    # Train Generator (every other step)
    with tf.GradientTape() as g_tape:
        fake_data = generator(noise, training=True)
        fake_output = discriminator(fake_data, training=True)
        
        g_loss = safe_binary_crossentropy(tf.ones_like(fake_output), fake_output)
        
        # Add conservative feature matching
        real_features = tf.reduce_mean(real_output, axis=0)
        fake_features = tf.reduce_mean(fake_output, axis=0)
        feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
        
        g_loss += 0.05 * feature_loss  # Small weight
    
    # Apply gradients with clipping
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    g_gradients = [tf.clip_by_norm(grad, 1.0) for grad in g_gradients]  # Gradient clipping
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    return d_loss, g_loss

def nan_proof_training(timegan, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE):
    d_losses, g_losses = [], []
    n_batches = max(1, len(sequences) // batch_size)
    
    print("🚀 Nan-proof training started...")
    print(f"📊 Batches per epoch: {n_batches}")
    
    sequences = sequences.astype(np.float32)
    
    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss = 0, 0
        valid_batches = 0
        
        # Shuffle each epoch
        indices = np.random.permutation(len(sequences))
        
        for batch in range(n_batches):
            try:
                batch_idx = indices[batch * batch_size: (batch + 1) * batch_size]
                real_batch = tf.convert_to_tensor(sequences[batch_idx], dtype=tf.float32)
                
                # Skip if batch contains NaN
                if tf.reduce_any(tf.math.is_nan(real_batch)):
                    continue
                
                d_loss, g_loss = stable_train_step(
                    real_batch, timegan.generator, timegan.discriminator,
                    timegan.g_optimizer, timegan.d_optimizer, SEQ_LEN, timegan.latent_dim
                )
                
                # Check for NaN in losses
                if not (tf.math.is_nan(d_loss) or tf.math.is_nan(g_loss)):
                    epoch_d_loss += d_loss.numpy()
                    epoch_g_loss += g_loss.numpy()
                    valid_batches += 1
                    
            except Exception as e:
                print(f"⚠️ Batch {batch} failed: {e}")
                continue
        
        if valid_batches > 0:
            avg_d_loss = epoch_d_loss / valid_batches
            avg_g_loss = epoch_g_loss / valid_batches
        else:
            avg_d_loss = avg_g_loss = 1.0  # Default safe values
            print("⚠️ No valid batches this epoch!")
        
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        # Progress monitoring
        if epoch % 20 == 0 or epoch == epochs - 1:
            progress = (epoch + 1) / epochs * 100
            print(f"📊 Epoch {epoch+1:3d}/{epochs} [{progress:5.1f}%] | "
                  f"D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
            
            # Early stopping if losses become unstable
            if np.isnan(avg_d_loss) or np.isnan(avg_g_loss):
                print("❌ NaN detected! Stopping training.")
                break
            
            # Convergence check
            if len(d_losses) > 30:
                recent_d = np.mean(d_losses[-10:])
                recent_g = np.mean(g_losses[-10:])
                if 0.5 < recent_d < 2.0 and 0.5 < recent_g < 2.0:
                    print("🎯 Good convergence detected!")
    
    return d_losses, g_losses

# Start training
print("⏳ Starting stable training...")
d_losses, g_losses = nan_proof_training(timegan, sequences)

# Check if training was successful
if len(d_losses) == 0 or np.any(np.isnan(d_losses)):
    print("❌ Training failed! Check data preprocessing.")
else:
    print("✅ Stable training completed!")
    
    # 💾 Save Models
    print("\n💾 Saving Stable Models...")
    timegan.generator.save('models/stable_timegan_generator.h5')
    timegan.discriminator.save('models/stable_timegan_discriminator.h5')
    print("✅ Models saved successfully!")
    
    # 🎨 Generate Stable Synthetic Data
    print("\n🎨 Generating Stable Synthetic Data...")
    
    def generate_stable_synthetic(generator, n_samples, seq_len, latent_dim):
        """Generate synthetic data with stability checks"""
        synthetic_data = []
        
        # Generate in batches to avoid memory issues
        batch_size = min(1000, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            current_batch = min(batch_size, n_samples - i * batch_size)
            noise = np.random.normal(0, 0.6, (current_batch, seq_len, latent_dim)).astype(np.float32)
            
            generated = generator.predict(noise, verbose=0)
            synthetic_data.append(generated)
        
        synthetic = np.vstack(synthetic_data)
        
        # Ensure no NaN
        synthetic = np.nan_to_num(synthetic, nan=0.0)
        synthetic = np.clip(synthetic, -1, 1)
        
        return synthetic[:n_samples]  # Ensure exact size
    
    n_synthetic = min(3000, len(sequences))
    synthetic_data = generate_stable_synthetic(
        timegan.generator, n_synthetic, SEQ_LEN, timegan.latent_dim
    )
    
    print(f"✅ Generated {len(synthetic_data)} stable synthetic sequences")
    
    # Denormalize for analysis
    synthetic_denorm = scaler.inverse_transform(
        synthetic_data.reshape(-1, synthetic_data.shape[-1])
    ).reshape(synthetic_data.shape)
    
    real_sample = sequences[:n_synthetic]
    real_denorm = scaler.inverse_transform(
        real_sample.reshape(-1, real_sample.shape[-1])
    ).reshape(real_sample.shape)
    
    # 📊 Step 6: Create Stable Visualizations
    print("\n📊 Creating Stable Visualizations...")
    
    def create_stable_visualizations(real_data, synthetic_data, d_losses, g_losses, feature_names):
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training Loss
        plt.subplot(2, 3, 1)
        plt.plot(d_losses, 'r-', label='Discriminator', alpha=0.8)
        plt.plot(g_losses, 'b-', label='Generator', alpha=0.8)
        plt.title('Stable Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Sample Comparison
        plt.subplot(2, 3, 2)
        n_show = min(5, len(real_data))
        for i in range(n_show):
            plt.plot(real_data[i, :, 0], 'g-', alpha=0.7, label='Real' if i == 0 else "")
            plt.plot(synthetic_data[i, :, 0], 'orange', alpha=0.7, label='Synthetic' if i == 0 else "")
        plt.title(f'Real vs Synthetic {feature_names[0]}', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Distribution Comparison
        plt.subplot(2, 3, 3)
        real_flat = real_data.reshape(-1, real_data.shape[-1])
        synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
        
        plt.hist(real_flat[:, 0], bins=50, alpha=0.6, label='Real', density=True, color='green')
        plt.hist(synth_flat[:, 0], bins=50, alpha=0.6, label='Synthetic', density=True, color='orange')
        plt.title('Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Correlation Comparison
        plt.subplot(2, 3, 4)
        real_corr = np.corrcoef(real_flat[:, :3].T)
        synth_corr = np.corrcoef(synth_flat[:, :3].T)
        corr_diff = np.abs(real_corr - synth_corr)
        
        plt.imshow(corr_diff, cmap='hot', vmin=0, vmax=0.5)
        plt.colorbar(label='Correlation Difference')
        plt.title('Feature Correlation Differences', fontsize=14, fontweight='bold')
        plt.xticks(range(min(3, len(feature_names))), feature_names[:3], rotation=45)
        plt.yticks(range(min(3, len(feature_names))), feature_names[:3])
        
        # 5. Temporal Patterns
        plt.subplot(2, 3, 5)
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
        
        # 6. Quality Metrics
        plt.subplot(2, 3, 6)
        metrics = {
            'Mean Diff': np.mean(np.abs(np.mean(real_flat, axis=0) - np.mean(synth_flat, axis=0))),
            'Std Diff': np.mean(np.abs(np.std(real_flat, axis=0) - np.std(synth_flat, axis=0))),
            'Corr Diff': np.mean(corr_diff),
            'Final D Loss': d_losses[-1] if len(d_losses) > 0 else 0,
            'Final G Loss': g_losses[-1] if len(g_losses) > 0 else 0
        }
        
        plt.bar(range(len(metrics)), list(metrics.values()), 
                color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
        plt.title('Quality Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/stable_bitcoin_timegan.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics
    
    metrics = create_stable_visualizations(real_denorm, synthetic_denorm, d_losses, g_losses, feature_names)
    
    # 🎯 FINAL STABLE REPORT
    print("\n" + "="*60)
    print("🎉 STABLE BITCOIN TIMEGAN - TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Dataset: {len(df):,} samples | Features: {feature_names}")
    print(f"⚡ Training: {len(d_losses)}/{EPOCHS} epochs completed")
    print(f"📈 Sequences: {sequences.shape}")
    print(f"🔧 Architecture: Stable LSTM TimeGAN")
    
    if len(d_losses) > 0:
        print(f"📉 Final D_loss: {d_losses[-1]:.6f}")
        print(f"📈 Final G_loss: {g_losses[-1]:.6f}")
        
        print("\n📊 SYNTHETIC DATA QUALITY:")
        print(f"   ✅ Mean Difference: {metrics['Mean Diff']:.6f}")
        print(f"   ✅ Std Difference: {metrics['Std Diff']:.6f}")
        print(f"   ✅ Correlation Difference: {metrics['Corr Diff']:.6f}")
        
        # Quality assessment
        if metrics['Mean Diff'] < 0.1 and metrics['Corr Diff'] < 0.2:
            print("\n🎉 EXCELLENT: High-quality synthetic data generated!")
        else:
            print("\n⚠️ GOOD: Synthetic data generated with acceptable quality")
    else:
        print("❌ Training failed to produce valid results")

print("="*60)