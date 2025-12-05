# Air Quality TimeGAN Training Script
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import numpy as np
import tensorflow as tf
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import sys

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
# Simple TimeGAN components
# ===========================================
class SimpleGenerator(tf.keras.Model):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.lstm1 = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='tanh')
        
    def call(self, z):
        x = self.dense1(z)
        x = self.lstm1(x)
        x = self.lstm2(x)
        return self.dense2(x)

class SimpleDiscriminator(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(hidden_dim)
        self.dense1 = tf.keras.layers.Dense(hidden_dim // 2, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        return self.dense2(x)

class SimpleTimeGAN:
    def __init__(self, config):
        self.config = config
        self.generator = SimpleGenerator(
            config['z_dim'], 
            config['hidden_dim'],
            config['feature_dim']
        )
        self.discriminator = SimpleDiscriminator(config['hidden_dim'])
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('lr_g', 1e-4))
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('lr_d', 5e-5))
        
    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        
        # Generate noise
        noise = tf.random.normal([batch_size, self.config['seq_len'], self.config['z_dim']])
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            # Generate fake data
            fake_data = self.generator(noise, training=True)
            
            # Discriminator predictions
            real_pred = self.discriminator(real_data, training=True)
            fake_pred = self.discriminator(fake_data, training=True)
            
            # Discriminator loss
            d_loss_real = tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_pred), real_pred
            )
            d_loss_fake = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_pred), fake_pred
            )
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
        
        # Apply gradients
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_grads = [tf.clip_by_norm(g, 1.0) for g in d_grads]
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        
        # Train generator
        with tf.GradientTape() as g_tape:
            fake_data = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_data, training=False)
            
            # Generator loss
            g_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(fake_pred), fake_pred
                )
            )
            
            # Add reconstruction loss for better quality
            if self.config.get('lambda_rec', 0) > 0:
                # Simple autoencoder-like loss
                rec_loss = tf.reduce_mean(tf.abs(real_data - fake_data))
                g_loss += self.config['lambda_rec'] * rec_loss
        
        # Apply gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_grads = [tf.clip_by_norm(g, 1.0) for g in g_grads]
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )
        
        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'real_pred': tf.reduce_mean(real_pred),
            'fake_pred': tf.reduce_mean(fake_pred)
        }
    
    def generate(self, n_samples):
        """Generate synthetic data"""
        noise = tf.random.normal([n_samples, self.config['seq_len'], self.config['z_dim']])
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

def create_dataset(data, batch_size=32, shuffle=True):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(data)))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def plot_training_history(history, output_dir):
    """Plot training history"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot losses
    axes[0, 0].plot(history['d_loss'], label='Discriminator Loss')
    axes[0, 0].plot(history['g_loss'], label='Generator Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot predictions
    axes[0, 1].plot(history['real_pred'], label='Real Predictions')
    axes[0, 1].plot(history['fake_pred'], label='Fake Predictions')
    axes[0, 1].set_title('Discriminator Predictions')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Prediction Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot sample comparison (only if available)
    if 'sample_real' in history and 'sample_fake' in history:
        try:
            axes[1, 0].plot(history['sample_real'][0, :, 0], label='Real', alpha=0.7)
            axes[1, 0].plot(history['sample_fake'][0, :, 0], label='Synthetic', alpha=0.7, linestyle='--')
            axes[1, 0].set_title('Sample Time Series (Feature 0)')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].text(0.5, 0.5, 'Sample data not available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Sample Time Series (Feature 0)')
    
    # Plot distribution comparison (only if available)
    if 'sample_real' in history and 'sample_fake' in history:
        try:
            real_flat = history['sample_real'][:, :, 0].flatten()
            fake_flat = history['sample_fake'][:, :, 0].flatten()
            
            axes[1, 1].hist(real_flat, bins=50, alpha=0.5, density=True, label='Real')
            axes[1, 1].hist(fake_flat, bins=50, alpha=0.5, density=True, label='Synthetic')
            axes[1, 1].set_title('Feature 0 Distribution')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Distribution data not available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature 0 Distribution')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    safe_print("[PLOT] Training history saved to {}".format(plot_path))

def validate_model(model, val_data, n_samples=100):
    """Validate the model"""
    # Generate synthetic data
    n_samples = min(n_samples, len(val_data))
    synthetic = model.generate(n_samples)
    
    # Basic statistics comparison
    def compute_stats(data):
        return {
            'mean': np.mean(data, axis=(0, 1)),
            'std': np.std(data, axis=(0, 1)),
            'min': np.min(data, axis=(0, 1)),
            'max': np.max(data, axis=(0, 1))
        }
    
    real_stats = compute_stats(val_data[:n_samples])
    synth_stats = compute_stats(synthetic)
    
    # Compute similarities
    metrics = {}
    for stat in ['mean', 'std']:
        real_val = real_stats[stat]
        synth_val = synth_stats[stat]
        
        # Correlation
        try:
            corr = np.corrcoef(real_val, synth_val)[0, 1]
            metrics[f'{stat}_correlation'] = float(corr) if not np.isnan(corr) else 0.0
        except:
            metrics[f'{stat}_correlation'] = 0.0
        
        # Mean absolute error
        mae = np.mean(np.abs(real_val - synth_val))
        metrics[f'{stat}_mae'] = float(mae)
    
    # Temporal pattern similarity (autocorrelation)
    def compute_autocorr(data, lag=3):
        acs = []
        for seq in data:
            for feat in range(min(3, data.shape[2])):
                series = seq[:, feat]
                if len(series) > lag:
                    try:
                        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        if not np.isnan(corr):
                            acs.append(np.abs(corr))
                    except:
                        continue
        return np.mean(acs) if acs else 0
    
    real_ac = compute_autocorr(val_data[:n_samples])
    synth_ac = compute_autocorr(synthetic)
    metrics['autocorr_similarity'] = 1 - np.abs(real_ac - synth_ac)
    
    # Overall score
    weights = {
        'mean_correlation': 0.4,
        'std_correlation': 0.4,
        'autocorr_similarity': 0.2
    }
    
    overall_score = 0
    for metric, weight in weights.items():
        if metric in metrics:
            overall_score += metrics[metric] * weight
    
    metrics['overall_score'] = float(overall_score)
    
    return metrics, synthetic

def save_model(model, checkpoint_dir, epoch, metrics):
    """Save model and metrics"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights
    generator_path = os.path.join(checkpoint_dir, 'generator_weights.h5')
    discriminator_path = os.path.join(checkpoint_dir, 'discriminator_weights.h5')
    
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

def train_model(config, train_data, val_data):
    """Main training function"""
    safe_print("\n" + "="*50)
    safe_print("TRAINING SIMPLE TIMEGAN")
    safe_print("="*50)
    
    # Initialize model
    model = SimpleTimeGAN(config)
    
    # Create dataset
    train_dataset = create_dataset(
        train_data, 
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    
    # Training history - initialize with required keys
    history = {
        'd_loss': [],
        'g_loss': [],
        'real_pred': [],
        'fake_pred': []
    }
    
    best_score = 0
    patience = 0
    
    # Training loop
    epochs = config.get('epochs', 50)
    for epoch in range(epochs):
        safe_print("\nEpoch {}/{}".format(epoch + 1, epochs))
        
        # Training
        epoch_losses = {'d_loss': [], 'g_loss': [], 'real_pred': [], 'fake_pred': []}
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_dataset):
            losses = model.train_step(batch)
            
            for key in epoch_losses.keys():
                epoch_losses[key].append(losses[key].numpy())
            
            batch_count = batch_idx + 1
            
            if batch_idx % 10 == 0:
                safe_print("  Batch {}: D_loss={:.4f}, G_loss={:.4f}".format(
                    batch_idx, losses['d_loss'].numpy(), losses['g_loss'].numpy()))
        
        # Update history - FIXED: Only update for keys that exist in both dictionaries
        for key in ['d_loss', 'g_loss', 'real_pred', 'fake_pred']:
            if key in epoch_losses and epoch_losses[key]:  # Check if key exists and list is not empty
                history[key].append(np.mean(epoch_losses[key]))
            else:
                history[key].append(0.0)  # Append default value if empty
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            safe_print("  Running validation...")
            metrics, synthetic = validate_model(model, val_data)
            
            if metrics:
                safe_print("  Validation Score: {:.4f}".format(metrics['overall_score']))
                safe_print("    Mean Correlation: {:.4f}".format(metrics['mean_correlation']))
                safe_print("    Std Correlation: {:.4f}".format(metrics['std_correlation']))
                
                # Save samples for plotting (only on specific epochs)
                should_save_samples = (epoch == 0) or ((epoch + 1) % 20 == 0)
                if should_save_samples:
                    history['sample_real'] = val_data[:5]
                    history['sample_fake'] = synthetic[:5]
                
                # Save best model
                if metrics['overall_score'] > best_score:
                    best_score = metrics['overall_score']
                    patience = 0
                    
                    save_model(
                        model, 
                        config['checkpoint_dir'], 
                        epoch, 
                        metrics
                    )
                    
                    safe_print("  [OK] New best model! Score: {:.4f}".format(best_score))
                else:
                    patience += 1
                    safe_print("  Patience: {}/{}".format(patience, config.get('patience', 10)))
                
                # Early stopping
                if patience >= config.get('patience', 10):
                    safe_print("\nEarly stopping at epoch {}".format(epoch))
                    break
            else:
                safe_print("  [WARNING] Validation metrics not available")
    
    safe_print("\n[SUCCESS] Training completed! Best score: {:.4f}".format(best_score))
    
    # Plot training history
    plot_training_history(history, config['checkpoint_dir'])
    
    return model

# ===========================================
# Main function
# ===========================================
def main():
    # Configuration
    
    # config = {
    #     'seq_len': 24,  # Will be updated from data
    #     'feature_dim': 15,  # Will be updated from data
    #     'hidden_dim': 128,
    #     'z_dim': 32,
    #     'batch_size': 64,
    #     'epochs':   300,
    #     'patience': 10,
    #     'lambda_rec': 0.1,
    #     'lr_g': 1e-4,
    #     'lr_d': 5e-5,
    #     'checkpoint_dir': 'checkpoints/air_quality'
    # }
    
    config = {
        'seq_len': 24,                    # Keep as per your data
        'feature_dim': 15,                # Keep as per your data
        'hidden_dim': 256,                # Increased from 128
        'z_dim': 64,                      # Increased from 32
        'batch_size': 64,                 # Optimal for your data size
        'epochs': 500,                    # Increased from 300
        'patience': 30,                   # Increased from 10
        'lambda_rec': 1.0,                # Increased from 0.1 (more emphasis on reconstruction)
        'lambda_gp': 10.0,                # NEW: Gradient penalty for stability
        'lr_g': 1e-4,                     # Generator learning rate
        'lr_d': 5e-5,                     # Discriminator learning rate (slightly lower)
        'checkpoint_dir': 'checkpoints/air_quality',
        
        # NEW HYPERPARAMETERS
        'n_critic': 5,                    # Train discriminator 5x more than generator
        'dropout_rate': 0.2,              # Regularization
        'gradient_clip': 1.0,             # Gradient clipping value
        'use_batch_norm': True,           # Use batch normalization
        'noise_std': 0.1,                 # Noise standard deviation
        'temperature': 0.5,               # For softmax/temperature scaling if needed
    }
    
    # config = {
    #     'seq_len': 24,  # Will be updated from data
    #     'feature_dim': 15,  # Will be updated from data
    #     'hidden_dim': 128,
    #     'z_dim': 32,
    #     'batch_size': 128,
    #     'epochs':   300,
    #     'patience': 10,
    #     'lambda_rec': 0.1,
    #     'lr_g': 1e-4,
    #     'lr_d': 5e-5,
    #     'checkpoint_dir': 'checkpoints/air_quality'
    # }
    
    # config = {
    #     'seq_len': 24,
    #     'feature_dim': 15,
    #     'hidden_dim': 256,  # Increased from 128
    #     'z_dim': 64,  # Increased from 32
    #     'batch_size': 64,  # Increased from 32
    #     'epochs': 300,  # Increased from 150
    #     'patience': 20,  # Increased from 10
    #     'lambda_rec': 1.0,  # Increased from 0.1
    #     'lambda_gp': 10.0,  # Gradient penalty weight
    #     'lambda_spectral': 0.5,  # Spectral loss weight
    #     'lambda_autocorr': 0.5,  # Autocorrelation loss weight
    #     'n_critic': 5,  # Train discriminator 5x more
    #     'lr_g': 5e-5,  # Adjusted learning rates
    #     'lr_d': 2e-5,
    #     'checkpoint_dir': 'checkpoints/air_quality'
    # }
    
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
        safe_print("  {}: {}".format(key, value))
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train model
    model = train_model(config, train_data, val_data)
    
    # Generate final synthetic dataset
    safe_print("\n" + "="*50)
    safe_print("GENERATING SYNTHETIC DATASET")
    safe_print("="*50)
    
    n_samples = 10000
    safe_print("Generating {} synthetic samples...".format(n_samples))
    synthetic_data = model.generate(min(n_samples, 5000))  # Limit for memory safety
    
    # Save synthetic data
    output_dir = 'outputs/synthetic_air_quality'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'synthetic_{timestamp}.npy')
    np.save(output_path, synthetic_data)
    
    safe_print("[SUCCESS] Synthetic data saved to: {}".format(output_path))
    safe_print("Synthetic data shape: {}".format(synthetic_data.shape))
    
    # Save some samples for inspection
    sample_path = os.path.join(output_dir, f'samples_{timestamp}.csv')
    sample_data = synthetic_data[:100].reshape(-1, synthetic_data.shape[2])
    np.savetxt(sample_path, sample_data, delimiter=',')
    safe_print("Sample data saved to: {}".format(sample_path))
    
    safe_print("\n" + "="*50)
    safe_print("PROCESS COMPLETED SUCCESSFULLY")
    safe_print("="*50)

if __name__ == "__main__":
    main()