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

print("🚀 Bitcoin TimeGAN - MEMORY OPTIMIZED VERSION")
print("=" * 70)

# Create directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== OPTIMIZED CONFIGURATION =====
CHUNK_SIZE = 1000000  # Process 1M rows at a time
SEQUENCES_PER_CHUNK = 5000  # Sequences from each chunk
EPOCHS_PER_CHUNK = 10  # Train on each chunk for fewer epochs
TOTAL_EPOCHS = 50  # Total training across all data
BATCH_SIZE = 32
HIDDEN_DIM = 32
LATENT_DIM = 8
SEQ_LEN = 24
# ===================``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````================

class MemoryOptimizedTimeGAN:
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

def load_data_in_chunks(file_path, chunk_size=CHUNK_SIZE):
    """Generator that yields data chunks"""
    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iterator):
        print(f"📦 Processing chunk {i+1}...")
        yield chunk

def preprocess_chunk(chunk, scaler=None, fit_scaler=False):
    """Preprocess a single chunk"""
    # Identify numeric columns
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    
    # Use Bitcoin columns if available, else first 5 numeric columns
    bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in bitcoin_cols if col in chunk.columns]
    
    if not available_cols:
        available_cols = numeric_cols[:5]
    
    data = chunk[available_cols].values
    data = np.nan_to_num(data)
    data = np.clip(data, -1e10, 1e10)
    
    # Scale the data
    if fit_scaler:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled, scaler, available_cols
    else:
        data_scaled = scaler.transform(data)
        return data_scaled, available_cols

def create_sequences_from_chunk(data, seq_len=SEQ_LEN, max_sequences=SEQUENCES_PER_CHUNK):
    """Create sequences from a single chunk"""
    n_possible = len(data) - seq_len
    if n_possible <= 0:
        return np.array([])
    
    n_sequences = min(max_sequences, n_possible)
    indices = np.random.choice(n_possible, n_sequences, replace=False)
    sequences = np.array([data[i:i+seq_len] for i in indices])
    
    return sequences

@tf.function
def train_step(generator, discriminator, g_opt, d_opt, real_batch, seq_len, latent_dim):
    """Single training step"""
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

def train_on_chunk(timegan, sequences, epochs=EPOCHS_PER_CHUNK):
    """Train model on a single chunk"""
    d_losses, g_losses = [], []
    batch_size = min(BATCH_SIZE, len(sequences))
    n_batches = max(1, len(sequences) // batch_size)
    
    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss = 0, 0
        indices = np.random.permutation(len(sequences))
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_indices = indices[start_idx:end_idx]
            real_batch = sequences[batch_indices]
            
            d_loss, g_loss = train_step(
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
        
        if epoch % 5 == 0:
            print(f"  📊 Chunk Epoch {epoch:2d} | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
    
    return d_losses, g_losses

# 🚀 MAIN TRAINING PIPELINE
print("\n🚀 Starting Memory-Optimized Training...")

# Initialize with first chunk to get feature dimensions
print("📊 Initializing with first chunk...")
chunk_generator = load_data_in_chunks('../data/bitcoin.csv')
first_chunk = next(chunk_generator)

# Preprocess first chunk and fit scaler
data_scaled, scaler, feature_names = preprocess_chunk(first_chunk, fit_scaler=True)
sequences = create_sequences_from_chunk(data_scaled)

print(f"📊 Feature names: {feature_names}")
print(f"📊 Sequence shape: {sequences.shape}")

# Initialize TimeGAN
timegan = MemoryOptimizedTimeGAN(SEQ_LEN, sequences.shape[2])

# Store all losses for plotting
all_d_losses, all_g_losses = [], []

# Train on first chunk
print(f"\n🏋️ Training on chunk 1...")
d_losses, g_losses = train_on_chunk(timegan, sequences)
all_d_losses.extend(d_losses)
all_g_losses.extend(g_losses)

# Continue with remaining chunks
chunk_num = 2
for chunk in chunk_generator:
    print(f"\n🏋️ Training on chunk {chunk_num}...")
    
    # Preprocess chunk using existing scaler
    data_scaled, _ = preprocess_chunk(chunk, scaler=scaler)
    sequences = create_sequences_from_chunk(data_scaled)
    
    if len(sequences) > 0:
        d_losses, g_losses = train_on_chunk(timegan, sequences)
        all_d_losses.extend(d_losses)
        all_g_losses.extend(g_losses)
    
    chunk_num += 1

print("✅ All chunks processed!")

# 🎨 Generate Synthetic Data
print("\n🎨 Generating Synthetic Data...")
def generate_synthetic(generator, n_samples, seq_len, latent_dim):
    noise = np.random.normal(0, 1, (n_samples, seq_len, latent_dim))
    return generator.predict(noise, verbose=0)

n_synthetic = 2000
synthetic_data = generate_synthetic(timegan.generator, n_synthetic, SEQ_LEN, timegan.latent_dim)

# Denormalize
synthetic_denorm = scaler.inverse_transform(
    synthetic_data.reshape(-1, synthetic_data.shape[-1])
).reshape(synthetic_data.shape)

# 💾 Save Everything
print("\n💾 Saving Results...")
timegan.generator.save('models/timegan_generator_optimized.h5')

# Save synthetic data
synthetic_flat = synthetic_denorm.reshape(-1, synthetic_denorm.shape[-1])
synthetic_df = pd.DataFrame(synthetic_flat, columns=feature_names)
synthetic_df.to_csv('results/synthetic_bitcoin_optimized.csv', index=False)

# 📊 Plot Results
print("\n📊 Creating Final Visualizations...")
plt.figure(figsize=(15, 10))

# Training loss
plt.subplot(2, 2, 1)
plt.plot(all_d_losses, 'r-', alpha=0.7, label='Discriminator')
plt.plot(all_g_losses, 'b-', alpha=0.7, label='Generator')
plt.title('Training Loss (All Chunks)')
plt.xlabel('Training Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample comparison
plt.subplot(2, 2, 2)
plt.plot(synthetic_denorm[0, :, 0], 'orange', label='Synthetic', linewidth=2)
plt.title(f'Synthetic {feature_names[0]}')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

# Distribution
plt.subplot(2, 2, 3)
synth_flat = synthetic_denorm.reshape(-1, synthetic_denorm.shape[-1])
plt.hist(synth_flat[:, 0], bins=50, alpha=0.7, color='orange', density=True)
plt.title('Synthetic Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Feature correlations
plt.subplot(2, 2, 4)
n_features = min(4, synthetic_denorm.shape[-1])
corr_matrix = np.corrcoef(synth_flat[:, :n_features].T)
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Synthetic Feature Correlations')
plt.xticks(range(n_features), feature_names[:n_features])
plt.yticks(range(n_features), feature_names[:n_features])

plt.tight_layout()
plt.savefig('results/plots/optimized_training.png', dpi=300, bbox_inches='tight')
plt.show()

# 🎯 Final Report
print("\n" + "="*60)
print("🎉 MEMORY-OPTIMIZED TIMEGAN TRAINING COMPLETE!")
print("="*60)
print(f"📊 Total chunks processed: {chunk_num}")
print(f"⚡ Total training epochs: {len(all_d_losses)}")
print(f"💾 Model saved: models/timegan_generator_optimized.h5")
print(f"💾 Synthetic data: {synthetic_df.shape}")
print(f"📈 Features: {feature_names}")
print("\n✅ Success! Efficiently processed large dataset!")