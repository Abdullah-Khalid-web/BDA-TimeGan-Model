# Air Quality TimeGAN Training Script with Improved Architecture
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import numpy as np
import tensorflow as tf
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# FIX: Force UTF-8 encoding for Windows
# ===========================================
def safe_print(message):
    """Safely print messages avoiding Unicode issues"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Replace Unicode characters with ASCII equivalents
        replacements = {
            '✅': '[SUCCESS]',
            '❌': '[FAILED]',
            '⚠️': '[WARNING]',
            '✓': '[OK]',
            '📊': '[PLOT]',
            '💾': '[SAVE]'
        }
        safe_message = message
        for uni_char, ascii_replacement in replacements.items():
            safe_message = safe_message.replace(uni_char, ascii_replacement)
        # Also handle unicode escape sequences
        safe_message = safe_message.replace('\u2705', '[SUCCESS]')
        safe_message = safe_message.replace('\u274c', '[FAILED]')
        safe_message = safe_message.replace('\u26a0', '[WARNING]')
        print(safe_message)

# ===========================================
# Improved TimeGAN Components
# ===========================================
class ImprovedGenerator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial projection
        self.dense_proj = tf.keras.layers.Dense(
            config['hidden_dim'] * 2,
            kernel_initializer='glorot_uniform'
        )
        
        # Multiple LSTM layers with residual connections
        self.lstm_layers = []
        for i in range(config.get('num_layers', 3)):
            lstm = tf.keras.layers.LSTM(
                config['hidden_dim'], 
                return_sequences=True,
                dropout=config.get('dropout_rate', 0.3),
                recurrent_dropout=config.get('dropout_rate', 0.3),
                kernel_initializer='glorot_uniform'
            )
            self.lstm_layers.append(lstm)
        
        # Attention mechanism
        self.use_attention = config.get('use_attention', True)
        if self.use_attention:
            self.attention = tf.keras.layers.Attention(use_scale=True)
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
        # Output layers
        self.dense1 = tf.keras.layers.Dense(
            config['hidden_dim'], 
            activation='relu',
            kernel_initializer='glorot_uniform'
        )
        self.dense2 = tf.keras.layers.Dense(
            config['feature_dim'], 
            activation='tanh',
            kernel_initializer='glorot_uniform'
        )
        
        # Skip connection projection
        self.skip_proj = tf.keras.layers.Dense(config['hidden_dim'])
        
    def call(self, z, training=False):
        x = self.dense_proj(z)
        
        # Apply LSTM layers with residual connections
        residual = self.skip_proj(z)
        for i, lstm_layer in enumerate(self.lstm_layers):
            x = lstm_layer(x, training=training)
            if i == 1 and self.config.get('use_residual', True):  # Add residual connection
                x = x + residual
        
        # Apply attention if enabled
        if self.use_attention and training:
            x = self.attention([x, x])
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Final projection
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x

class ImprovedDiscriminator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Spectral normalization if enabled
        if config.get('use_spectral_norm', True):
            self.dense1 = tf.keras.layers.Dense(
                config['hidden_dim'],
                kernel_initializer='glorot_uniform'
            )
            self.spectral_norm = tf.keras.layers.Lambda(
                lambda x: tf.linalg.normalize(x, axis=-1)[0] * tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
            )
        else:
            self.dense1 = tf.keras.layers.Dense(
                config['hidden_dim'],
                kernel_initializer='glorot_uniform'
            )
        
        # LSTM layers
        self.lstm1 = tf.keras.layers.LSTM(
            config['hidden_dim'], 
            return_sequences=True,
            dropout=config.get('dropout_rate', 0.3),
            recurrent_dropout=config.get('dropout_rate', 0.3),
            kernel_initializer='glorot_uniform'
        )
        self.lstm2 = tf.keras.layers.LSTM(
            config['hidden_dim'],
            dropout=config.get('dropout_rate', 0.3),
            recurrent_dropout=config.get('dropout_rate', 0.3),
            kernel_initializer='glorot_uniform'
        )
        
        # Output layers
        self.dense2 = tf.keras.layers.Dense(
            config['hidden_dim'] // 2, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config.get('weight_decay', 1e-5))
        )
        self.dense3 = tf.keras.layers.Dense(
            1, 
            activation='linear',  # Linear for Wasserstein loss
            kernel_regularizer=tf.keras.regularizers.l2(config.get('weight_decay', 1e-5))
        )
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, x, training=False):
        # Initial projection with spectral normalization if enabled
        if self.config.get('use_spectral_norm', True):
            x = self.dense1(x)
            x = self.spectral_norm(x)
        else:
            x = self.dense1(x)
        
        # LSTM layers
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output layers
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x

class ImprovedTimeGAN:
    def __init__(self, config):
        self.config = config
        self.generator = ImprovedGenerator(config)
        self.discriminator = ImprovedDiscriminator(config)
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('lr_g', 5e-5),
            beta_1=config.get('beta1', 0.5),
            beta_2=config.get('beta2', 0.9)
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('lr_d', 2e-5),
            beta_1=config.get('beta1', 0.5),
            beta_2=config.get('beta2', 0.9)
        )
        
        # Learning rate scheduler
        self.lr_scheduler_g = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.get('lr_g', 5e-5),
            decay_steps=1000,
            decay_rate=config.get('lr_decay', 0.98)
        )
        self.lr_scheduler_d = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.get('lr_d', 2e-5),
            decay_steps=1000,
            decay_rate=config.get('lr_decay', 0.98)
        )
        
        # Loss tracking
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        self.gp_metric = tf.keras.metrics.Mean(name='gradient_penalty')
        self.rec_metric = tf.keras.metrics.Mean(name='reconstruction_loss')
        
    def gradient_penalty(self, real_data, fake_data):
        """Compute gradient penalty for Wasserstein loss"""
        batch_size = tf.shape(real_data)[0]
        
        # Generate random epsilon
        epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        
        # Interpolated samples
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        # Compute gradients
        gradients = tape.gradient(pred, interpolated)
        
        # Compute gradient penalty
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        return gradient_penalty
    
    def compute_embedding_loss(self, real_data, synthetic_data):
        """Compute embedding loss to preserve temporal relationships"""
        # Use mean pooling as a simple embedding
        real_embed = tf.reduce_mean(real_data, axis=1)
        synth_embed = tf.reduce_mean(synthetic_data, axis=1)
        
        # Cosine similarity loss
        norm_real = tf.nn.l2_normalize(real_embed, axis=-1)
        norm_synth = tf.nn.l2_normalize(synth_embed, axis=-1)
        cosine_sim = tf.reduce_sum(norm_real * norm_synth, axis=-1)
        
        # We want high similarity, so loss = 1 - similarity
        embedding_loss = 1.0 - tf.reduce_mean(cosine_sim)
        
        return embedding_loss
    
    def compute_autocorrelation_loss(self, real_data, synthetic_data, max_lag=3):
        """Compute autocorrelation loss to preserve temporal patterns"""
        batch_size = tf.shape(real_data)[0]
        seq_len = tf.shape(real_data)[1]
        n_features = tf.shape(real_data)[2]
        
        real_ac = []
        synth_ac = []
        
        for lag in range(1, max_lag + 1):
            # Ensure we have enough sequence length for this lag
            if seq_len > lag:
                # Real autocorrelation
                real_lag = real_data[:, lag:, :]
                real_base = real_data[:, :-lag, :]
                
                # Flatten for correlation computation
                real_lag_flat = tf.reshape(real_lag, [-1, n_features])
                real_base_flat = tf.reshape(real_base, [-1, n_features])
                
                # Compute correlation per feature
                for feat in range(n_features):
                    real_corr = self._compute_correlation(
                        real_base_flat[:, feat], 
                        real_lag_flat[:, feat]
                    )
                    if not tf.math.is_nan(real_corr):
                        real_ac.append(tf.abs(real_corr))
                
                # Synthetic autocorrelation
                synth_lag = synthetic_data[:, lag:, :]
                synth_base = synthetic_data[:, :-lag, :]
                
                synth_lag_flat = tf.reshape(synth_lag, [-1, n_features])
                synth_base_flat = tf.reshape(synth_base, [-1, n_features])
                
                for feat in range(n_features):
                    synth_corr = self._compute_correlation(
                        synth_base_flat[:, feat], 
                        synth_lag_flat[:, feat]
                    )
                    if not tf.math.is_nan(synth_corr):
                        synth_ac.append(tf.abs(synth_corr))
        
        if real_ac and synth_ac:
            real_ac_mean = tf.reduce_mean(tf.stack(real_ac))
            synth_ac_mean = tf.reduce_mean(tf.stack(synth_ac))
            ac_loss = tf.abs(real_ac_mean - synth_ac_mean)
        else:
            ac_loss = tf.constant(0.0)
        
        return ac_loss
    
    def _compute_correlation(self, x, y):
        """Compute correlation between two tensors"""
        mx = tf.reduce_mean(x)
        my = tf.reduce_mean(y)
        
        xm = x - mx
        ym = y - my
        
        r_num = tf.reduce_sum(xm * ym)
        r_den = tf.sqrt(tf.reduce_sum(xm * xm) * tf.reduce_sum(ym * ym))
        
        r = r_num / (r_den + 1e-8)
        
        return r
    
    @tf.function
    def train_step(self, real_data, epoch):
        batch_size = tf.shape(real_data)[0]
        
        # Generate noise with annealing
        current_noise_std = self.config.get('noise_std_start', 0.3) * tf.pow(
            self.config.get('noise_decay', 0.99), 
            tf.cast(epoch, tf.float32)
        )
        current_noise_std = tf.maximum(current_noise_std, self.config.get('noise_std_end', 0.05))
        
        noise = tf.random.normal(
            [batch_size, self.config['seq_len'], self.config['z_dim']], 
            stddev=current_noise_std
        )
        
        # Train discriminator more frequently
        n_critic = self.config.get('n_critic', 3)
        d_loss_total = 0.0
        gp_total = 0.0
        
        for _ in range(n_critic):
            with tf.GradientTape() as d_tape:
                # Generate fake data
                fake_data = self.generator(noise, training=True)
                
                # Discriminator predictions
                real_pred = self.discriminator(real_data, training=True)
                fake_pred = self.discriminator(fake_data, training=True)
                
                # Wasserstein loss
                d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                
                # Gradient penalty
                gp = self.gradient_penalty(real_data, fake_data)
                d_loss += self.config.get('lambda_gp', 10.0) * gp
                
                # Add L2 regularization if specified
                if self.config.get('weight_decay', 0) > 0:
                    d_loss += tf.add_n([
                        self.config['weight_decay'] * tf.nn.l2_loss(v)
                        for v in self.discriminator.trainable_variables
                    ])
            
            # Apply discriminator gradients
            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            if self.config.get('gradient_clip', 0.5) > 0:
                d_grads = [tf.clip_by_norm(g, self.config['gradient_clip']) for g in d_grads]
            self.d_optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )
            
            d_loss_total += d_loss
            gp_total += gp
        
        # Average discriminator loss over n_critic steps
        d_loss_avg = d_loss_total / n_critic
        gp_avg = gp_total / n_critic
        
        # Train generator
        with tf.GradientTape() as g_tape:
            # Generate fake data
            fake_data = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_data, training=False)
            
            # Adversarial loss (negative for Wasserstein)
            g_loss_adv = -tf.reduce_mean(fake_pred)
            
            # Reconstruction loss - CRITICAL for data fidelity
            rec_loss = tf.reduce_mean(tf.abs(real_data - fake_data))
            
            # Embedding loss for temporal relationships
            emb_loss = self.compute_embedding_loss(real_data, fake_data)
            
            # Autocorrelation loss for temporal patterns
            ac_loss = self.compute_autocorrelation_loss(real_data, fake_data)
            
            # Total generator loss with weighted components
            g_loss = (
                self.config.get('lambda_adv', 1.0) * g_loss_adv +
                self.config.get('lambda_rec', 10.0) * rec_loss +  # INCREASED from 0.1
                self.config.get('lambda_emb', 100.0) * emb_loss +  # NEW
                self.config.get('lambda_ac', 0.5) * ac_loss  # NEW
            )
            
            # Add L2 regularization if specified
            if self.config.get('weight_decay', 0) > 0:
                g_loss += tf.add_n([
                    self.config['weight_decay'] * tf.nn.l2_loss(v)
                    for v in self.generator.trainable_variables
                ])
        
        # Apply generator gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        if self.config.get('gradient_clip', 0.5) > 0:
            g_grads = [tf.clip_by_norm(g, self.config['gradient_clip']) for g in g_grads]
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )
        
        # Update metrics
        self.g_loss_metric(g_loss)
        self.d_loss_metric(d_loss_avg)
        self.gp_metric(gp_avg)
        self.rec_metric(rec_loss)
        
        return {
            'd_loss': d_loss_avg,
            'g_loss': g_loss,
            'rec_loss': rec_loss,
            'emb_loss': emb_loss,
            'ac_loss': ac_loss,
            'gp': gp_avg,
            'real_pred': tf.reduce_mean(real_pred),
            'fake_pred': tf.reduce_mean(fake_pred)
        }
    
    def generate(self, n_samples, temperature=1.0):
        """Generate synthetic data"""
        noise = tf.random.normal([n_samples, self.config['seq_len'], self.config['z_dim']])
        
        # Apply temperature scaling if needed
        if temperature != 1.0:
            noise = noise * temperature
        
        return self.generator(noise, training=False).numpy()

# ===========================================
# Helper functions
# ===========================================
def load_data(data_dir):
    """Load processed data"""
    train_path = os.path.join(data_dir, 'train.npy')
    val_path = os.path.join(data_dir, 'val.npy')
    
    if not os.path.exists(train_path):
        safe_print("[FAILED] Training data not found at {}".format(train_path))
        safe_print("Current working directory: {}".format(os.getcwd()))
        safe_print("Looking for: {}".format(os.abspath(train_path)))
        return None, None
    
    try:
        train_data = np.load(train_path)
        val_data = np.load(val_path) if os.path.exists(val_path) else train_data[:100]
        
        safe_print("[SUCCESS] Loaded data:")
        safe_print("  Train: {}".format(train_data.shape))
        safe_print("  Val: {}".format(val_data.shape))
        
        return train_data, val_data
    except Exception as e:
        safe_print("[FAILED] Error loading data: {}".format(e))
        return None, None

def normalize_data(data):
    """Normalize data to [-1, 1] range for better GAN training"""
    data_min = np.min(data, axis=(0, 1), keepdims=True)
    data_max = np.max(data, axis=(0, 1), keepdims=True)
    
    # Avoid division by zero
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    # Normalize to [-1, 1]
    normalized = 2.0 * (data - data_min) / data_range - 1.0
    
    # Save normalization parameters
    norm_params = {
        'min': data_min,
        'max': data_max,
        'range': data_range
    }
    
    return normalized, norm_params

def denormalize_data(data, norm_params):
    """Denormalize data back to original range"""
    return (data + 1.0) * norm_params['range'] / 2.0 + norm_params['min']

def create_dataset(data, batch_size=32, shuffle=True, buffer_size=1000):
    """Create TensorFlow dataset with improved pipeline"""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(buffer_size, len(data)))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def plot_training_history(history, output_dir, config):
    """Plot comprehensive training history"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot losses
    axes[0, 0].plot(history['d_loss'], label='Discriminator Loss', alpha=0.8)
    axes[0, 0].plot(history['g_loss'], label='Generator Loss', alpha=0.8)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot component losses
    if 'rec_loss' in history and history['rec_loss']:
        axes[0, 1].plot(history['rec_loss'], label='Reconstruction Loss', alpha=0.8)
    if 'emb_loss' in history and history['emb_loss']:
        axes[0, 1].plot(history['emb_loss'], label='Embedding Loss', alpha=0.8)
    if 'ac_loss' in history and history['ac_loss']:
        axes[0, 1].plot(history['ac_loss'], label='Auto-correlation Loss', alpha=0.8)
    axes[0, 1].set_title('Generator Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot gradient penalty
    if 'gp' in history and history['gp']:
        axes[0, 2].plot(history['gp'], label='Gradient Penalty', color='red', alpha=0.8)
        axes[0, 2].set_title('Gradient Penalty')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Penalty')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].remove()
    
    # Plot predictions
    if 'real_pred' in history and 'fake_pred' in history:
        axes[1, 0].plot(history['real_pred'], label='Real Predictions', alpha=0.7)
        axes[1, 0].plot(history['fake_pred'], label='Fake Predictions', alpha=0.7)
        axes[1, 0].set_title('Discriminator Predictions')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Prediction Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot sample comparison
    if 'sample_real' in history and 'sample_fake' in history:
        try:
            real_sample = history['sample_real'][0, :, 0]
            fake_sample = history['sample_fake'][0, :, 0]
            
            axes[1, 1].plot(real_sample, label='Real', alpha=0.7, linewidth=2)
            axes[1, 1].plot(fake_sample, label='Synthetic', alpha=0.7, linewidth=2, linestyle='--')
            axes[1, 1].set_title('Sample Time Series (Feature 0)')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Sample data not available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Sample Time Series (Feature 0)')
    
    # Plot distribution comparison
    if 'sample_real' in history and 'sample_fake' in history:
        try:
            real_flat = history['sample_real'][:, :, 0].flatten()
            fake_flat = history['sample_fake'][:, :, 0].flatten()
            
            axes[1, 2].hist(real_flat, bins=50, alpha=0.5, density=True, label='Real', color='blue')
            axes[1, 2].hist(fake_flat, bins=50, alpha=0.5, density=True, label='Synthetic', color='red')
            axes[1, 2].set_title('Feature 0 Distribution')
            axes[1, 2].set_xlabel('Value')
            axes[1, 2].set_ylabel('Density')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        except:
            axes[1, 2].text(0.5, 0.5, 'Distribution data not available', 
                          ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature 0 Distribution')
    
    plt.suptitle(f'TimeGAN Training History\nHidden Dim: {config["hidden_dim"]}, Z Dim: {config["z_dim"]}, Lambda Rec: {config.get("lambda_rec", 10.0)}', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    safe_print("[PLOT] Training history saved to {}".format(plot_path))

def validate_model(model, val_data, n_samples=100):
    """Validate the model with comprehensive metrics"""
    # Generate synthetic data
    n_samples = min(n_samples, len(val_data))
    synthetic = model.generate(n_samples)
    
    # Basic statistics comparison
    def compute_stats(data):
        return {
            'mean': np.mean(data, axis=(0, 1)),
            'std': np.std(data, axis=(0, 1)),
            'min': np.min(data, axis=(0, 1)),
            'max': np.max(data, axis=(0, 1)),
            'median': np.median(data, axis=(0, 1))
        }
    
    real_stats = compute_stats(val_data[:n_samples])
    synth_stats = compute_stats(synthetic)
    
    # Compute similarities
    metrics = {}
    
    # Mean statistics
    for stat in ['mean', 'std', 'median']:
        real_val = real_stats[stat]
        synth_val = synth_stats[stat]
        
        # Correlation
        try:
            corr = np.corrcoef(real_val, synth_val)[0, 1]
            metrics[f'{stat}_correlation'] = float(corr) if not np.isnan(corr) else 0.0
        except:
            metrics[f'{stat}_correlation'] = 0.0
        
        # Mean absolute error (normalized)
        mae = np.mean(np.abs(real_val - synth_val))
        norm_factor = np.std(real_val) if np.std(real_val) > 0 else 1.0
        metrics[f'{stat}_mae'] = float(mae / norm_factor)
    
    # Range similarity
    real_range = real_stats['max'] - real_stats['min']
    synth_range = synth_stats['max'] - synth_stats['min']
    range_similarity = 1.0 - np.mean(np.abs(real_range - synth_range) / (real_range + 1e-8))
    metrics['range_similarity'] = float(np.clip(range_similarity, 0, 1))
    
    # Temporal pattern similarity (autocorrelation)
    def compute_autocorr(data, max_lag=3):
        acs = []
        for seq in data:
            for feat in range(min(5, data.shape[2])):  # Check first 5 features
                series = seq[:, feat]
                for lag in range(1, min(max_lag + 1, len(series))):
                    try:
                        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        if not np.isnan(corr):
                            acs.append(np.abs(corr))
                    except:
                        continue
        return np.mean(acs) if acs else 0
    
    real_ac = compute_autocorr(val_data[:n_samples])
    synth_ac = compute_autocorr(synthetic)
    metrics['autocorr_similarity'] = float(1 - np.abs(real_ac - synth_ac))
    
    # Cross-correlation between features
    def compute_crosscorr(data):
        # Compute average cross-correlation across time steps
        n_features = data.shape[2]
        ccs = []
        for seq in data:
            # Reshape to (time, features)
            seq_2d = seq.reshape(seq.shape[0], n_features)
            try:
                corr_matrix = np.corrcoef(seq_2d.T)
                # Take average of off-diagonal elements
                if corr_matrix.size > 1:
                    ccs.append(np.mean(np.abs(corr_matrix[np.triu_indices(n_features, k=1)])))
            except:
                continue
        return np.mean(ccs) if ccs else 0
    
    real_cc = compute_crosscorr(val_data[:n_samples])
    synth_cc = compute_crosscorr(synthetic)
    metrics['crosscorr_similarity'] = float(1 - np.abs(real_cc - synth_cc))
    
    # Overall score with weighted components
    weights = {
        'mean_correlation': 0.25,
        'std_correlation': 0.25,
        'range_similarity': 0.15,
        'autocorr_similarity': 0.20,
        'crosscorr_similarity': 0.15
    }
    
    overall_score = 0
    for metric, weight in weights.items():
        if metric in metrics:
            overall_score += metrics[metric] * weight
    
    metrics['overall_score'] = float(overall_score)
    
    # Store sample for visualization
    sample_real = val_data[:min(5, len(val_data))]
    sample_fake = synthetic[:5]
    
    return metrics, synthetic, sample_real, sample_fake

def save_model(model, checkpoint_dir, epoch, metrics):
    """Save model and metrics"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights
    generator_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch:04d}.h5')
    discriminator_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch:04d}.h5')
    
    model.generator.save_weights(generator_path)
    model.discriminator.save_weights(discriminator_path)
    
    # Save configuration
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model.config, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(checkpoint_dir, f'metrics_epoch_{epoch:04d}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    safe_print("  [OK] Model saved to {}".format(checkpoint_dir))
    
    return generator_path, discriminator_path

def load_latest_checkpoint(model, checkpoint_dir):
    """Load latest checkpoint if exists"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find latest generator checkpoint
    gen_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('generator_epoch_') and f.endswith('.h5')]
    if not gen_files:
        return None
    
    # Extract epoch numbers and find latest
    epochs = []
    for f in gen_files:
        try:
            epoch_num = int(f.split('_')[2].split('.')[0])
            epochs.append(epoch_num)
        except:
            continue
    
    if not epochs:
        return None
    
    latest_epoch = max(epochs)
    
    # Load weights
    generator_path = os.path.join(checkpoint_dir, f'generator_epoch_{latest_epoch:04d}.h5')
    discriminator_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{latest_epoch:04d}.h5')
    
    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        model.generator.load_weights(generator_path)
        model.discriminator.load_weights(discriminator_path)
        
        # Load metrics
        metrics_path = os.path.join(checkpoint_dir, f'metrics_epoch_{latest_epoch:04d}.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = None
        
        safe_print("[SUCCESS] Loaded checkpoint from epoch {}".format(latest_epoch))
        return latest_epoch, metrics
    
    return None

def train_model(config, train_data, val_data, start_epoch=0):
    """Main training function"""
    safe_print("\n" + "="*60)
    safe_print("TRAINING IMPROVED TIMEGAN")
    safe_print("="*60)
    
    # Normalize data
    safe_print("\nNormalizing data...")
    train_data_norm, norm_params = normalize_data(train_data)
    val_data_norm, _ = normalize_data(val_data)
    
    # Save normalization parameters
    norm_path = os.path.join(config['checkpoint_dir'], 'normalization_params.pkl')
    with open(norm_path, 'wb') as f:
        pickle.dump(norm_params, f)
    safe_print("[OK] Normalization parameters saved")
    
    # Initialize model
    model = ImprovedTimeGAN(config)
    
    # Try to load latest checkpoint
    checkpoint_info = load_latest_checkpoint(model, config['checkpoint_dir'])
    if checkpoint_info:
        last_epoch, last_metrics = checkpoint_info
        start_epoch = last_epoch + 1
        safe_print("Resuming training from epoch {}".format(start_epoch))
        if last_metrics and 'overall_score' in last_metrics:
            best_score = last_metrics['overall_score']
            safe_print("Previous best score: {:.4f}".format(best_score))
        else:
            best_score = 0
    else:
        best_score = 0
    
    # Create dataset
    train_dataset = create_dataset(
        train_data_norm, 
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        buffer_size=min(2000, len(train_data))
    )
    
    # Training history
    history = {
        'd_loss': [],
        'g_loss': [],
        'rec_loss': [],
        'emb_loss': [],
        'ac_loss': [],
        'gp': [],
        'real_pred': [],
        'fake_pred': [],
        'val_score': []
    }
    
    patience = 0
    no_improvement_count = 0
    best_epoch = start_epoch
    
    # Training loop
    epochs = config.get('epochs', 500)
    for epoch in range(start_epoch, epochs):
        safe_print("\nEpoch {}/{}".format(epoch + 1, epochs))
        
        # Warmup phase: adjust learning rate
        if epoch < config.get('warmup_epochs', 50):
            warmup_factor = (epoch + 1) / config['warmup_epochs']
            model.g_optimizer.learning_rate = config['lr_g'] * warmup_factor
            model.d_optimizer.learning_rate = config['lr_d'] * warmup_factor
        
        # Training
        epoch_losses = {
            'd_loss': [], 'g_loss': [], 'rec_loss': [], 
            'emb_loss': [], 'ac_loss': [], 'gp': [],
            'real_pred': [], 'fake_pred': []
        }
        
        batch_count = 0
        for batch_idx, batch in enumerate(train_dataset):
            losses = model.train_step(batch, epoch)
            
            for key in losses.keys():
                if key in epoch_losses:
                    epoch_losses[key].append(losses[key].numpy())
            
            batch_count = batch_idx + 1
            
            if batch_idx % 10 == 0:
                safe_print("  Batch {}: D={:.4f}, G={:.4f}, Rec={:.4f}".format(
                    batch_idx, 
                    losses['d_loss'].numpy(), 
                    losses['g_loss'].numpy(),
                    losses['rec_loss'].numpy()
                ))
        
        # Update history
        for key in epoch_losses.keys():
            if epoch_losses[key]:
                history[key].append(np.mean(epoch_losses[key]))
            else:
                history[key].append(0.0)
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            safe_print("  Running validation...")
            metrics, synthetic, sample_real, sample_fake = validate_model(model, val_data_norm)
            
            if metrics:
                val_score = metrics['overall_score']
                history['val_score'].append(val_score)
                
                safe_print("  Validation Score: {:.4f}".format(val_score))
                safe_print("    Mean Correlation: {:.4f}".format(metrics['mean_correlation']))
                safe_print("    Std Correlation: {:.4f}".format(metrics['std_correlation']))
                safe_print("    Auto-corr Similarity: {:.4f}".format(metrics['autocorr_similarity']))
                
                # Save samples for plotting (only on specific epochs)
                should_save_samples = (epoch == 0) or ((epoch + 1) % 20 == 0)
                if should_save_samples:
                    history['sample_real'] = sample_real
                    history['sample_fake'] = sample_fake
                
                # Save best model
                if val_score > best_score:
                    best_score = val_score
                    best_epoch = epoch
                    patience = 0
                    no_improvement_count = 0
                    
                    save_model(model, config['checkpoint_dir'], epoch, metrics)
                    
                    safe_print("  [SUCCESS] New best model! Score: {:.4f}".format(best_score))
                else:
                    patience += 1
                    no_improvement_count += 1
                    safe_print("  Patience: {}/{}".format(patience, config.get('patience', 40)))
                
                # Learning rate scheduling based on validation plateau
                if no_improvement_count >= 10:
                    # Reduce learning rates
                    model.g_optimizer.learning_rate *= 0.5
                    model.d_optimizer.learning_rate *= 0.5
                    safe_print("  [WARNING] Reducing learning rates due to plateau")
                    no_improvement_count = 0
                
                # Early stopping
                if patience >= config.get('patience', 40):
                    safe_print("\n[WARNING] Early stopping at epoch {} (best epoch: {}, score: {:.4f})".format(
                        epoch, best_epoch, best_score))
                    break
            else:
                safe_print("  [WARNING] Validation metrics not available")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            safe_print("  Saving checkpoint...")
            save_model(model, config['checkpoint_dir'], epoch, metrics if 'metrics' in locals() else {})
    
    safe_print("\n" + "="*60)
    safe_print("[SUCCESS] Training completed!")
    safe_print("  Best score: {:.4f} at epoch {}".format(best_score, best_epoch))
    safe_print("  Final validation score: {:.4f}".format(history['val_score'][-1] if history['val_score'] else 0))
    safe_print("="*60)
    
    # Plot training history
    plot_training_history(history, config['checkpoint_dir'], config)
    
    # Load best model for final generation
    best_gen_path = os.path.join(config['checkpoint_dir'], f'generator_epoch_{best_epoch:04d}.h5')
    if os.path.exists(best_gen_path):
        model.generator.load_weights(best_gen_path)
        safe_print("[OK] Loaded best model from epoch {}".format(best_epoch))
    
    return model, norm_params

# ===========================================
# Main function
# ===========================================
def main():
    # IMPROVED CONFIGURATION
    config = {
        # Architecture
        'seq_len': 24,                     # Will be updated from data
        'feature_dim': 15,                 # Will be updated from data
        'hidden_dim': 256,                 # Increased capacity
        'z_dim': 100,                      # Increased latent dimension
        'num_layers': 3,                   # Increased from 2 to 3 LSTM layers
        
        # Training parameters
        'batch_size': 128,                 # Increased batch size
        'epochs': 500,                     
        'patience': 40,                    # Increased patience
        
        # Loss weights (CRITICAL IMPROVEMENTS)
        'lambda_rec': 10.0,                # SIGNIFICANTLY INCREASED - was 0.1!
        'lambda_adv': 1.0,                 # Adversarial loss weight
        'lambda_emb': 100.0,               # NEW: Embedding loss for time-series
        'lambda_ac': 0.5,                  # NEW: Autocorrelation loss
        'lambda_gp': 10.0,                 # Gradient penalty weight
        
        # Regularization
        'dropout_rate': 0.3,               # Increased dropout
        'gradient_clip': 0.5,              # Gradient clipping
        'weight_decay': 1e-5,              # L2 regularization
        
        # Optimizer settings
        'lr_g': 5e-5,                      # Lower learning rate for generator
        'lr_d': 2e-5,                      # Lower learning rate for discriminator
        'beta1': 0.5,                      # Adam beta1
        'beta2': 0.9,                      # Adam beta2
        'lr_decay': 0.98,                  # Learning rate decay
        
        # Training strategy
        'n_critic': 3,                     # Train discriminator 3x more
        'warmup_epochs': 50,               # Warmup phase
        'noise_std_start': 0.3,            # Initial noise std
        'noise_std_end': 0.05,             # Final noise std
        'noise_decay': 0.99,               # Noise decay rate
        
        # Advanced features
        'use_attention': True,             # Add attention mechanism
        'use_layer_norm': True,            # Layer norm for time-series
        'use_residual': True,              # Residual connections
        'use_spectral_norm': True,         # Spectral normalization
        
        'checkpoint_dir': 'checkpoints/air_quality_improved'
    }
    
    # Load data
    data_dir = 'data/processed/air_quality'
    safe_print("\nLoading data from {}...".format(data_dir))
    safe_print("Current directory: {}".format(os.getcwd()))
    
    train_data, val_data = load_data(data_dir)
    
    if train_data is None:
        safe_print("[FAILED] Could not load training data!")
        # Try alternative path
        alt_path = '../data/processed/air_quality'
        safe_print("Trying alternative path: {}".format(alt_path))
        train_data, val_data = load_data(alt_path)
        
        if train_data is None:
            safe_print("[FAILED] Could not load data from any path!")
            sys.exit(1)
    
    # Update config from data
    config['seq_len'] = train_data.shape[1]
    config['feature_dim'] = train_data.shape[2]
    
    safe_print("\nConfiguration:")
    for key, value in config.items():
        safe_print("  {:20}: {}".format(key, value))
    
    safe_print("\nData Statistics:")
    safe_print("  Train shape: {}".format(train_data.shape))
    safe_print("  Validation shape: {}".format(val_data.shape))
    safe_print("  Min value: {:.4f}".format(np.min(train_data)))
    safe_print("  Max value: {:.4f}".format(np.max(train_data)))
    safe_print("  Mean value: {:.4f}".format(np.mean(train_data)))
    safe_print("  Std value: {:.4f}".format(np.std(train_data)))
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train model
    model, norm_params = train_model(config, train_data, val_data)
    
    # Generate final synthetic dataset
    safe_print("\n" + "="*60)
    safe_print("GENERATING SYNTHETIC DATASET")
    safe_print("="*60)
    
    n_samples = 10000
    safe_print("Generating {} synthetic samples...".format(n_samples))
    
    # Generate in batches to avoid memory issues
    batch_size = 1000
    synthetic_batches = []
    
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        safe_print("  Generating batch {}/{}...".format(i // batch_size + 1, (n_samples + batch_size - 1) // batch_size))
        
        synthetic_batch = model.generate(current_batch)
        synthetic_batches.append(synthetic_batch)
    
    # Concatenate all batches
    synthetic_data = np.concatenate(synthetic_batches, axis=0)
    
    # Denormalize synthetic data
    safe_print("Denormalizing synthetic data...")
    synthetic_data_denorm = denormalize_data(synthetic_data, norm_params)
    
    # Save synthetic data
    output_dir = 'outputs/synthetic_air_quality_improved'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'synthetic_{timestamp}.npy')
    np.save(output_path, synthetic_data_denorm)
    
    # Save normalized version too
    output_path_norm = os.path.join(output_dir, f'synthetic_norm_{timestamp}.npy')
    np.save(output_path_norm, synthetic_data)
    
    safe_print("[SUCCESS] Synthetic data saved to:")
    safe_print("  Raw: {}".format(output_path))
    safe_print("  Normalized: {}".format(output_path_norm))
    safe_print("Synthetic data shape: {}".format(synthetic_data_denorm.shape))
    
    # Save some samples for inspection
    sample_path = os.path.join(output_dir, f'samples_{timestamp}.csv')
    sample_data = synthetic_data_denorm[:100].reshape(-1, synthetic_data_denorm.shape[2])
    np.savetxt(sample_path, sample_data, delimiter=',')
    safe_print("Sample data saved to: {}".format(sample_path))
    
    # Save configuration
    config_path = os.path.join(output_dir, f'config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save normalization parameters
    norm_path = os.path.join(output_dir, f'norm_params_{timestamp}.pkl')
    with open(norm_path, 'wb') as f:
        pickle.dump(norm_params, f)
    
    # Generate a quick comparison plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot sample time series
        real_sample = train_data[0, :, 0]
        synth_sample = synthetic_data_denorm[0, :, 0]
        
        axes[0, 0].plot(real_sample, label='Real', alpha=0.8, linewidth=2)
        axes[0, 0].plot(synth_sample, label='Synthetic', alpha=0.8, linewidth=2, linestyle='--')
        axes[0, 0].set_title('Sample Time Series Comparison (Feature 0)')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot distributions
        real_flat = train_data[:, :, 0].flatten()[:1000]
        synth_flat = synthetic_data_denorm[:, :, 0].flatten()[:1000]
        
        axes[0, 1].hist(real_flat, bins=50, alpha=0.5, density=True, label='Real', color='blue')
        axes[0, 1].hist(synth_flat, bins=50, alpha=0.5, density=True, label='Synthetic', color='red')
        axes[0, 1].set_title('Feature 0 Distribution Comparison')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot means across features
        real_means = np.mean(train_data, axis=(0, 1))
        synth_means = np.mean(synthetic_data_denorm, axis=(0, 1))
        
        axes[1, 0].bar(np.arange(len(real_means)) - 0.2, real_means, width=0.4, label='Real', alpha=0.7)
        axes[1, 0].bar(np.arange(len(synth_means)) + 0.2, synth_means, width=0.4, label='Synthetic', alpha=0.7)
        axes[1, 0].set_title('Feature-wise Means Comparison')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot stds across features
        real_stds = np.std(train_data, axis=(0, 1))
        synth_stds = np.std(synthetic_data_denorm, axis=(0, 1))
        
        axes[1, 1].bar(np.arange(len(real_stds)) - 0.2, real_stds, width=0.4, label='Real', alpha=0.7)
        axes[1, 1].bar(np.arange(len(synth_stds)) + 0.2, synth_stds, width=0.4, label='Synthetic', alpha=0.7)
        axes[1, 1].set_title('Feature-wise Standard Deviations Comparison')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Real vs Synthetic Data Comparison', fontsize=14)
        plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, f'comparison_{timestamp}.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        safe_print("[PLOT] Comparison visualization saved to {}".format(comparison_path))
    except Exception as e:
        safe_print("[WARNING] Could not generate comparison plot: {}".format(e))
    
    safe_print("\n" + "="*60)
    safe_print("PROCESS COMPLETED SUCCESSFULLY")
    safe_print("="*60)
    safe_print("\nSummary:")
    safe_print("  - Trained with improved TimeGAN architecture")
    safe_print("  - Used reconstruction loss weight: {}".format(config['lambda_rec']))
    safe_print("  - Generated {} synthetic samples".format(n_samples))
    safe_print("  - All outputs saved to: {}".format(output_dir))
    safe_print("  - Checkpoints saved to: {}".format(config['checkpoint_dir']))
    safe_print("="*60)

if __name__ == "__main__":
    main()