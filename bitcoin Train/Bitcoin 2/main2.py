import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 ENHANCED BITCOIN TIMEGAN - HIGH ACCURACY VERSION")
print("=" * 70)

# Create directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ===== OPTIMIZED CONFIGURATION =====
MAX_SEQUENCES = 8000  # Balanced for quality
EPOCHS = 400  # More epochs for convergence
BATCH_SIZE = 128  # Larger batch for stability
HIDDEN_DIM = 128  # Increased capacity
LATENT_DIM = 32  # Larger latent space
SEQ_LEN = 24
LEARNING_RATE = 0.0001  # Lower learning rate
GRADIENT_PENALTY_WEIGHT = 10.0
FEATURE_MATCHING_WEIGHT = 0.1
# ==================================

# 🔥 Step 1: Load and Preprocess Data with Enhanced Cleaning - FIXED WITH SAMPLE SIZE
print("\n📊 Step 1: Loading and Enhanced Preprocessing...")

def load_and_preprocess_enhanced(file_path, sample_size=None, random_state=42):
    """Enhanced data loading and preprocessing with sampling option"""
    try:
        print("📁 Loading Bitcoin dataset...")
        
        # Load with or without sampling
        if sample_size:
            print(f"🎯 Sampling {sample_size:,} rows from dataset...")
            
            # Get total rows (excluding header)
            total_rows = sum(1 for line in open(file_path)) - 1
            
            if sample_size >= total_rows:
                print("⚠️ Sample size larger than dataset, loading full data...")
                df = pd.read_csv(file_path)
            else:
                # Sample random rows efficiently
                skip_rows = np.random.choice(total_rows, total_rows - sample_size, replace=False)
                skip_rows = [x + 1 for x in skip_rows]  # +1 to skip header
                df = pd.read_csv(file_path, skiprows=skip_rows)
        else:
            print("✅ Loading full dataset...")
            df = pd.read_csv(file_path)
        
        # Enhanced column detection
        if df.shape[1] >= 5:
            if 'Open' in df.columns:
                bitcoin_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                bitcoin_cols = list(df.columns[:5])
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(df.columns[5:])
        else:
            bitcoin_cols = list(df.columns)
        
        print(f"✅ Using columns: {bitcoin_cols}")
        print(f"📊 Loaded data shape: {df.shape}")
        
        # Enhanced data cleaning
        data = df[bitcoin_cols].copy()
        
        # Remove outliers using IQR
        for col in bitcoin_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Enhanced normalization - StandardScaler for better temporal patterns
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_scaled = data_scaled.astype(np.float32)
        
        print(f"📊 Final data shape: {data_scaled.shape}")
        print(f"🎯 Data range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
        
        return data_scaled, scaler, bitcoin_cols
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

# 🔥 CHOOSE YOUR SAMPLE SIZE HERE:
data_scaled, scaler, feature_names = load_and_preprocess_enhanced('../data/bitcoin.csv')
print(f"\n⏱️ Step 2: Creating Enhanced Sequences...")

def create_optimized_sequences(data, seq_len=SEQ_LEN, max_seq=MAX_SEQUENCES):
    """Create sequences with overlap for more training data"""
    n_possible = len(data) - seq_len
    n_sequences = min(max_seq, n_possible)
    
    # Use overlapping sequences for more diverse training
    step_size = max(1, (len(data) - seq_len) // n_sequences)
    sequences = []
    
    for i in range(0, n_possible, step_size):
        if len(sequences) >= n_sequences:
            break
        sequences.append(data[i:i+seq_len])
    
    sequences = np.array(sequences, dtype=np.float32)
    
    print(f"✅ Created {len(sequences)} sequences")
    print(f"📊 Sequence shape: {sequences.shape}")
    
    return sequences

sequences = create_optimized_sequences(data_scaled)

# 🧠 Step 3: Build High-Accuracy TimeGAN Model
print("\n🧠 Step 3: Building High-Accuracy TimeGAN Model...")

class HighAccuracyTimeGAN:
    def __init__(self, seq_len, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.generator = self._build_advanced_generator()
        self.discriminator = self._build_advanced_discriminator()
        self.embedder = self._build_embedder()  # For feature matching
        
        # Use separate learning rates without schedule initially
        self.g_optimizer = keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(LEARNING_RATE * 0.5, beta_1=0.5)
        
        print(f"✅ High-Accuracy TimeGAN built!")
        print(f"   Generator: {self.generator.count_params():,} params")
        print(f"   Discriminator: {self.discriminator.count_params():,} params")
        print(f"   Embedder: {self.embedder.count_params():,} params")
    
    def _build_advanced_generator(self):
        """Enhanced generator with residual connections"""
        inputs = keras.Input(shape=(self.seq_len, self.latent_dim))
        
        # First LSTM layer
        x = layers.LSTM(self.hidden_dim, return_sequences=True, 
                       dropout=0.1, recurrent_dropout=0.1)(inputs)
        x = layers.BatchNormalization()(x)
        
        # Second LSTM layer with residual
        residual = x
        x = layers.LSTM(self.hidden_dim, return_sequences=True,
                       dropout=0.1, recurrent_dropout=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])  # Residual connection
        
        # Third LSTM layer
        x = layers.LSTM(self.hidden_dim // 2, return_sequences=True,
                       dropout=0.1, recurrent_dropout=0.1)(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Softmax(axis=1)(attention)
        x = layers.Multiply()([x, attention])
        
        # Output layer
        outputs = layers.Dense(self.n_features, activation='linear')(x)
        
        return keras.Model(inputs, outputs, name="Advanced_Generator")
    
    def _build_advanced_discriminator(self):
        """Enhanced discriminator with multiple outputs"""
        inputs = keras.Input(shape=(self.seq_len, self.n_features))
        
        x = layers.LSTM(self.hidden_dim, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2)(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(self.hidden_dim // 2, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Multiple outputs for sequence discrimination
        sequence_output = layers.LSTM(self.hidden_dim // 4, return_sequences=False)(x)
        
        # Final discrimination
        outputs = layers.Dense(1, activation='sigmoid')(sequence_output)
        
        return keras.Model(inputs, outputs, name="Advanced_Discriminator")
    
    def _build_embedder(self):
        """Embedder for feature matching loss"""
        inputs = keras.Input(shape=(self.seq_len, self.n_features))
        
        x = layers.LSTM(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.LSTM(self.hidden_dim // 2, return_sequences=True)(x)
        x = layers.LSTM(self.hidden_dim // 4, return_sequences=False)(x)
        outputs = layers.Dense(self.hidden_dim // 8, activation='relu')(x)
        
        return keras.Model(inputs, outputs, name="Embedder")

timegan = HighAccuracyTimeGAN(SEQ_LEN, sequences.shape[2])

# 🏋️ Step 4: Enhanced Training with Multiple Loss Components - FIXED VERSION
print(f"\n🏋️ Step 4: Training for {EPOCHS} epochs...")

@tf.function
def high_accuracy_train_step(timegan, real_batch):
    batch_size = tf.shape(real_batch)[0]
    
    # Generate noise
    noise = tf.random.normal((batch_size, SEQ_LEN, timegan.latent_dim), 
                           stddev=0.5, dtype=tf.float32)
    
    # Train Discriminator
    with tf.GradientTape() as d_tape:
        fake_data = timegan.generator(noise, training=True)
        
        real_output = timegan.discriminator(real_batch, training=True)
        fake_output = timegan.discriminator(fake_data, training=True)
        
        # Discriminator losses
        d_loss_real = tf.reduce_mean(
            keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
        d_loss_fake = tf.reduce_mean(
            keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        # Gradient penalty
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
        interpolated = epsilon * real_batch + (1 - epsilon) * fake_data
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_output = timegan.discriminator(interpolated, training=True)
        
        gradients = gp_tape.gradient(interpolated_output, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        d_loss += GRADIENT_PENALTY_WEIGHT * gradient_penalty
    
    d_grads = d_tape.gradient(d_loss, timegan.discriminator.trainable_variables)
    timegan.d_optimizer.apply_gradients(zip(d_grads, timegan.discriminator.trainable_variables))
    
    # Train Generator
    with tf.GradientTape() as g_tape:
        fake_data = timegan.generator(noise, training=True)
        fake_output = timegan.discriminator(fake_data, training=True)
        
        # Generator adversarial loss
        g_loss_adv = tf.reduce_mean(
            keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
        
        # Feature matching loss
        real_embeddings = timegan.embedder(real_batch, training=True)
        fake_embeddings = timegan.embedder(fake_data, training=True)
        feature_matching_loss = tf.reduce_mean(
            tf.abs(real_embeddings - fake_embeddings))
        
        # Statistical moment matching
        real_mean = tf.reduce_mean(real_batch, axis=[0, 1])
        fake_mean = tf.reduce_mean(fake_data, axis=[0, 1])
        real_std = tf.math.reduce_std(real_batch, axis=[0, 1])
        fake_std = tf.math.reduce_std(fake_data, axis=[0, 1])
        
        moment_loss = (tf.reduce_mean(tf.abs(real_mean - fake_mean)) +
                      tf.reduce_mean(tf.abs(real_std - fake_std)))
        
        # Combined generator loss
        g_loss = (g_loss_adv + 
                 FEATURE_MATCHING_WEIGHT * feature_matching_loss +
                 0.1 * moment_loss)
    
    g_grads = g_tape.gradient(g_loss, timegan.generator.trainable_variables)
    timegan.g_optimizer.apply_gradients(zip(g_grads, timegan.generator.trainable_variables))
    
    return d_loss, g_loss, g_loss_adv, feature_matching_loss

def high_accuracy_train(timegan, sequences, epochs=EPOCHS, batch_size=BATCH_SIZE):
    d_losses, g_losses, g_adv_losses, fm_losses = [], [], [], []
    n_batches = max(1, len(sequences) // batch_size)
    
    print("🚀 High-accuracy training started...")
    print(f"📊 Batches per epoch: {n_batches}")
    
    # FIXED: Simple learning rate decay without complex scheduling
    initial_g_lr = LEARNING_RATE
    initial_d_lr = LEARNING_RATE * 0.5
    
    best_g_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Simple learning rate decay every 50 epochs
        if epoch > 0 and epoch % 50 == 0:
            new_g_lr = initial_g_lr * (0.8 ** (epoch // 50))
            new_d_lr = initial_d_lr * (0.8 ** (epoch // 50))
            timegan.g_optimizer.learning_rate.assign(new_g_lr)
            timegan.d_optimizer.learning_rate.assign(new_d_lr)
        
        epoch_d_loss, epoch_g_loss, epoch_g_adv, epoch_fm = 0, 0, 0, 0
        
        # Shuffle sequences
        indices = np.random.permutation(len(sequences))
        
        for batch in range(n_batches):
            batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
            real_batch = tf.convert_to_tensor(sequences[batch_indices], dtype=tf.float32)
            
            d_loss, g_loss, g_adv, fm_loss = high_accuracy_train_step(timegan, real_batch)
            
            epoch_d_loss += d_loss.numpy()
            epoch_g_loss += g_loss.numpy()
            epoch_g_adv += g_adv.numpy()
            epoch_fm += fm_loss.numpy()
        
        # Average losses
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        avg_g_adv = epoch_g_adv / n_batches
        avg_fm = epoch_fm / n_batches
        
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        g_adv_losses.append(avg_g_adv)
        fm_losses.append(avg_fm)
        
        # Enhanced monitoring
        if epoch % 20 == 0 or epoch == epochs - 1:
            progress = (epoch + 1) / epochs * 100
            current_g_lr = timegan.g_optimizer.learning_rate.numpy()
            current_d_lr = timegan.d_optimizer.learning_rate.numpy()
            
            print(f"📊 Epoch {epoch+1:3d}/{epochs} [{progress:5.1f}%]")
            print(f"   D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
            print(f"   G_adv: {avg_g_adv:.4f} | FM_loss: {avg_fm:.4f}")
            print(f"   G_LR: {current_g_lr:.6f} | D_LR: {current_d_lr:.6f}")
        
        # Early stopping with patience
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            patience_counter = 0
            # Save best model
            timegan.generator.save('models/timegan_generator_best.h5')
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > 100:
            print(f"🎯 Early stopping at epoch {epoch+1}")
            break
    
    return d_losses, g_losses, g_adv_losses, fm_losses

# Start training
d_losses, g_losses, g_adv_losses, fm_losses = high_accuracy_train(timegan, sequences)

# 🎨 Step 5: Generate High-Fidelity Synthetic Data
print("\n🎨 Generating High-Fidelity Synthetic Data...")

def generate_high_fidelity_synthetic(generator, n_samples, seq_len, latent_dim):
    """Generate synthetic data with multiple strategies"""
    
    # Strategy 1: Standard generation
    noise_std = [0.3, 0.5, 0.7]  # Multiple noise levels
    synthetic_parts = []
    
    for std in noise_std:
        n_part = n_samples // len(noise_std)
        noise = np.random.normal(0, std, (n_part, seq_len, latent_dim)).astype(np.float32)
        generated = generator.predict(noise, verbose=0)
        synthetic_parts.append(generated)
    
    synthetic_data = np.vstack(synthetic_parts)
    
    # Add small noise to increase diversity
    noise_factor = 0.01
    synthetic_data += np.random.normal(0, noise_factor, synthetic_data.shape).astype(np.float32)
    
    return synthetic_data

n_synthetic = min(3000, len(sequences))
synthetic_data = generate_high_fidelity_synthetic(timegan.generator, n_synthetic, SEQ_LEN, timegan.latent_dim)

# Denormalize
synthetic_denorm = scaler.inverse_transform(
    synthetic_data.reshape(-1, synthetic_data.shape[-1])
).reshape(synthetic_data.shape)

real_sample = sequences[:n_synthetic]
real_denorm = scaler.inverse_transform(
    real_sample.reshape(-1, real_sample.shape[-1])
).reshape(real_sample.shape)

# 📊 Step 6: Comprehensive Evaluation
print("\n📊 Step 6: Comprehensive Model Evaluation...")

def evaluate_synthetic_quality(real_data, synthetic_data, feature_names):
    """Comprehensive evaluation of synthetic data quality"""
    
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    metrics = {}
    
    # Statistical moments
    metrics['Mean_MAE'] = mean_absolute_error(
        np.mean(real_flat, axis=0), np.mean(synth_flat, axis=0))
    metrics['Std_MAE'] = mean_absolute_error(
        np.std(real_flat, axis=0), np.std(synth_flat, axis=0))
    
    # Correlation preservation
    real_corr = np.corrcoef(real_flat.T)
    synth_corr = np.corrcoef(synth_flat.T)
    metrics['Correlation_MAE'] = mean_absolute_error(real_corr, synth_corr)
    
    # Distribution similarity (Wasserstein distance approximation)
    from scipy.stats import wasserstein_distance
    wasserstein_dists = []
    for i in range(min(5, real_flat.shape[1])):
        w_dist = wasserstein_distance(real_flat[:, i], synth_flat[:, i])
        wasserstein_dists.append(w_dist)
    metrics['Avg_Wasserstein'] = np.mean(wasserstein_dists)
    
    # Temporal pattern preservation
    real_temporal = np.mean(real_data, axis=0)  # Average over sequences
    synth_temporal = np.mean(synthetic_data, axis=0)
    metrics['Temporal_MAE'] = mean_absolute_error(real_temporal, synth_temporal)
    
    print("🎯 SYNTHETIC DATA QUALITY METRICS:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.6f}")
    
    return metrics

quality_metrics = evaluate_synthetic_quality(real_denorm, synthetic_denorm, feature_names)

# 📈 Step 7: Simplified Visualization with 9 Clear Graphs
print("\n📈 Creating Simplified Visualizations (9 Graphs)...")

def create_simplified_visualizations(real_data, synthetic_data, d_losses, g_losses, 
                                   feature_names, quality_metrics):
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(25, 20))
    
    # 1. Training Loss Progress
    plt.subplot(3, 3, 1)
    epochs_range = range(len(d_losses))
    plt.plot(epochs_range, d_losses, 'r-', label='Discriminator Loss', linewidth=2)
    plt.plot(epochs_range, g_losses, 'b-', label='Generator Loss', linewidth=2)
    plt.title('Training Loss Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Real vs Synthetic Samples (First Feature)
    plt.subplot(3, 3, 2)
    n_samples_show = 4
    time_steps = range(real_data.shape[1])
    
    for i in range(n_samples_show):
        plt.plot(time_steps, real_data[i, :, 0], 'g-', alpha=0.7, 
                linewidth=2, label='Real' if i == 0 else "")
        plt.plot(time_steps, synthetic_data[i, :, 0], 'orange', alpha=0.7,
                linewidth=2, label='Synthetic' if i == 0 else "")
    
    plt.title(f'Real vs Synthetic {feature_names[0]}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Feature Distributions Comparison
    plt.subplot(3, 3, 3)
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
    
    # Plot first feature distribution
    plt.hist(real_flat[:, 0], bins=50, alpha=0.6, label='Real', 
             color='green', density=True)
    plt.hist(synth_flat[:, 0], bins=50, alpha=0.6, label='Synthetic', 
             color='orange', density=True)
    plt.title(f'{feature_names[0]} Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Correlation Matrix - Real Data
    plt.subplot(3, 3, 4)
    real_corr = np.corrcoef(real_flat[:, :5].T)
    im1 = plt.imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im1, label='Correlation')
    plt.title('Real Data Correlations', fontsize=14, fontweight='bold')
    plt.xticks(range(len(feature_names[:5])), feature_names[:5], rotation=45)
    plt.yticks(range(len(feature_names[:5])), feature_names[:5])
    
    # 5. Correlation Matrix - Synthetic Data
    plt.subplot(3, 3, 5)
    synth_corr = np.corrcoef(synth_flat[:, :5].T)
    im2 = plt.imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im2, label='Correlation')
    plt.title('Synthetic Data Correlations', fontsize=14, fontweight='bold')
    plt.xticks(range(len(feature_names[:5])), feature_names[:5], rotation=45)
    plt.yticks(range(len(feature_names[:5])), feature_names[:5])
    
    # 6. Temporal Pattern Comparison
    plt.subplot(3, 3, 6)
    real_mean = np.mean(real_data, axis=0)
    synth_mean = np.mean(synthetic_data, axis=0)
    
    # Plot first 2 features
    for i in range(min(2, real_data.shape[2])):
        plt.plot(real_mean[:, i], label=f'Real {feature_names[i]}', 
                color=f'C{i}', linewidth=3)
        plt.plot(synth_mean[:, i], label=f'Synthetic {feature_names[i]}', 
                color=f'C{i}', linestyle='--', linewidth=2)
    
    plt.title('Average Temporal Patterns', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Feature Relationship Scatter Plot
    plt.subplot(3, 3, 7)
    if len(feature_names) >= 2:
        plt.scatter(real_flat[:1000, 0], real_flat[:1000, 1], 
                   alpha=0.6, label='Real', s=20, color='green')
        plt.scatter(synth_flat[:1000, 0], synth_flat[:1000, 1], 
                   alpha=0.6, label='Synthetic', s=20, color='orange')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Feature Relationship', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 8. Quality Metrics Comparison
    plt.subplot(3, 3, 8)
    metric_names = ['Mean Error', 'Std Error', 'Correlation Error', 'Temporal Error']
    metric_values = [
        quality_metrics.get('Mean_MAE', 0),
        quality_metrics.get('Std_MAE', 0),
        quality_metrics.get('Correlation_MAE', 0),
        quality_metrics.get('Temporal_MAE', 0)
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)
    plt.title('Synthetic Data Quality Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Error Value (Lower = Better)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 9. Overall Quality Score
    plt.subplot(3, 3, 9)
    overall_quality = 1 - np.mean(metric_values) / 0.5  # Normalize
    overall_quality = max(0, min(1, overall_quality))
    
    plt.barh(['Quality Score'], [overall_quality * 100], 
             color='green' if overall_quality > 0.7 else 'orange', alpha=0.8)
    plt.xlim(0, 100)
    plt.title('Overall Synthetic Data Quality', fontsize=14, fontweight='bold')
    plt.xlabel('Quality Score (%)')
    
    # Add quality rating text
    if overall_quality > 0.8:
        rating = "EXCELLENT"
        color = "darkgreen"
    elif overall_quality > 0.6:
        rating = "GOOD"
        color = "orange"
    else:
        rating = "NEEDS IMPROVEMENT"
        color = "red"
    
    plt.text(50, 0, f'{rating}', ha='center', va='center', 
             fontsize=16, fontweight='bold', color=color,
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('results/plots/simplified_bitcoin_timegan.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_quality

# Create the simplified visualizations
overall_quality = create_simplified_visualizations(
    real_denorm, synthetic_denorm, d_losses, g_losses, 
    feature_names, quality_metrics
)

# 🎯 Final Summary Report
print("\n" + "="*60)
print("🎉 TIMEGAN TRAINING COMPLETE - SIMPLIFIED REPORT")
print("="*60)
print(f"📊 Dataset: {data_scaled.shape[0]:,} samples")
print(f"⚡ Training: {len(d_losses)} epochs completed")
print(f"📈 Sequences: {sequences.shape[0]} sequences")
print(f"🎯 Overall Quality: {overall_quality:.1%}")
print(f"📉 Final D Loss: {d_losses[-1]:.4f}")
print(f"📈 Final G Loss: {g_losses[-1]:.4f}")
print("\n📋 VISUALIZATION SUMMARY:")
print("   1. Training Loss - Model convergence")
print("   2. Sample Comparison - Real vs Synthetic examples")
print("   3. Distribution - Data spread comparison")
print("   4. Real Correlations - Feature relationships in real data")
print("   5. Synthetic Correlations - Feature relationships in synthetic data")
print("   6. Temporal Patterns - Time series behavior")
print("   7. Feature Relationship - How features relate to each other")
print("   8. Quality Metrics - Statistical accuracy measurements")
print("   9. Overall Quality - Final synthetic data quality score")