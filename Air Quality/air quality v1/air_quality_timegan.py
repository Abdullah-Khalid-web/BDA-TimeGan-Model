# Air Quality TimeGAN Implementation
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
regularizers = keras.regularizers
import numpy as np
import os

class AirQualityEmbedder(tf.keras.Model):
    """Embedder specifically designed for air quality patterns"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Use list comprehension instead of appending to lists
        self.conv1 = layers.Conv1D(
            hidden_dim // 2, kernel_size=3, padding='same',
            activation='relu', kernel_regularizer=regularizers.l2(1e-5)
        )
        
        # Create bidirectional LSTMs
        self.bilstms = []
        for i in range(num_layers):
            bilstm = layers.Bidirectional(
                layers.LSTM(hidden_dim // 4, return_sequences=True,
                           dropout=dropout, recurrent_dropout=dropout,
                           kernel_regularizer=regularizers.l2(1e-5)),
                name=f'air_bilstm_{i}'
            )
            self.bilstms.append(bilstm)
        
        # Attention for important time steps
        self.attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=hidden_dim // 8
        )
        
        # Final dense layer
        self.dense = layers.Dense(hidden_dim, activation='tanh')
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, x, training=False):
        # Conv for local pollution patterns
        x = self.conv1(x)
        
        # Bidirectional LSTMs
        for bilstm in self.bilstms:
            x = bilstm(x, training=training)
        
        # Attention mechanism
        attn_output = self.attention(x, x, training=training)
        x = layers.Concatenate()([x, attn_output])
        
        # Final transformation
        x = self.dense(x)
        x = self.layer_norm(x)
        
        return x

class AirQualityGenerator(tf.keras.Model):
    """Generator for air quality time series"""
    def __init__(self, z_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        # Initial projection
        self.dense_in = layers.Dense(hidden_dim, activation='relu')
        
        # Temporal convolutional layers
        self.conv_layers = []
        for i in range(2):
            conv = layers.Conv1DTranspose(
                hidden_dim, kernel_size=3, padding='same',
                activation='relu', kernel_regularizer=regularizers.l2(1e-5)
            )
            self.conv_layers.append(conv)
        
        # LSTM layers for temporal coherence
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.LSTM(
                hidden_dim, return_sequences=True,
                dropout=0.1, recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(1e-5)
            )
            self.lstms.append(lstm)
        
        # Output layers with skip connections
        self.dense_out = layers.Dense(hidden_dim, activation='tanh')
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, z, training=False):
        x = self.dense_in(z)
        
        # Convolutional processing with skip connections
        for conv in self.conv_layers:
            x_residual = x
            x = conv(x)
            x = layers.Add()([x, x_residual])
        
        # LSTM processing
        for lstm in self.lstms:
            x = lstm(x, training=training)
        
        x = self.dense_out(x)
        x = self.layer_norm(x)
        
        return x

class AirQualityDiscriminator(tf.keras.Model):
    """Discriminator for air quality sequences"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Feature extraction with multiple kernel sizes
        self.conv_layers = []
        kernel_sizes = [3, 5, 7]
        
        for i, kernel_size in enumerate(kernel_sizes):
            conv = layers.Conv1D(
                64 * (2 ** i), kernel_size=kernel_size,
                padding='same', activation='leaky_relu',
                kernel_regularizer=regularizers.l2(1e-5)
            )
            self.conv_layers.append(conv)
        
        # Multi-head attention for global patterns
        self.attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=64
        )
        
        # Bidirectional LSTM for temporal context
        self.bilstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True,
                       kernel_regularizer=regularizers.l2(1e-5))
        )
        
        # Classification head
        self.global_avg = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(256, activation='leaky_relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(128, activation='leaky_relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
        
    def call(self, x, training=False):
        # Multi-scale convolutional feature extraction
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x, training=training)
            conv_outputs.append(conv_out)
        
        # Combine multi-scale features
        x = layers.Concatenate()(conv_outputs)
        
        # Attention mechanism
        attn_output = self.attention(x, x, training=training)
        x = layers.Add()([x, attn_output])
        
        # Temporal modeling
        x = self.bilstm(x, training=training)
        
        # Classification
        x = self.global_avg(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        return self.output_layer(x)

class AirQualitySupervisor(tf.keras.Model):
    """Supervisor network for air quality data"""
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.LSTM(
                hidden_dim, return_sequences=True,
                dropout=0.1, recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(1e-5)
            )
            self.lstms.append(lstm)
        
        self.dense = layers.Dense(hidden_dim, activation='tanh')
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, x, training=False):
        for lstm in self.lstms:
            x = lstm(x, training=training)
        
        x = self.dense(x)
        x = self.layer_norm(x)
        
        return x

class AirQualityRecovery(tf.keras.Model):
    """Recovery network for air quality data"""
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.LSTM(
                hidden_dim, return_sequences=True,
                dropout=0.1, recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(1e-5)
            )
            self.lstms.append(lstm)
        
        # Multiple output heads for different feature groups
        self.dense_heads = []
        n_heads = 3
        head_dim = output_dim // n_heads
        
        for i in range(n_heads):
            dense = layers.Dense(head_dim, activation='linear')
            self.dense_heads.append(dense)
        
        self.final_dense = layers.Dense(output_dim, activation='linear')
        
    def call(self, x, training=False):
        for lstm in self.lstms:
            x = lstm(x, training=training)
        
        # Multiple heads then combine
        head_outputs = []
        for dense in self.dense_heads:
            head_outputs.append(dense(x))
        
        x = layers.Concatenate()(head_outputs)
        x = self.final_dense(x)
        
        return x

class AirQualityTimeGAN:
    """Complete TimeGAN for air quality data"""
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.embedder = AirQualityEmbedder(
            config['feature_dim'], config['hidden_dim']
        )
        self.generator = AirQualityGenerator(
            config['z_dim'], config['hidden_dim']
        )
        self.discriminator = AirQualityDiscriminator(config['hidden_dim'])
        self.supervisor = AirQualitySupervisor(config['hidden_dim'])
        self.recovery = AirQualityRecovery(
            config['hidden_dim'], config['feature_dim']
        )
        
        # Build models
        self._build_models()
        
    def _build_models(self):
        """Build models with proper input shapes"""
        batch_size = self.config.get('batch_size', 32)
        seq_len = self.config['seq_len']
        
        # Test shapes with sample data
        test_input = tf.random.normal([batch_size, seq_len, self.config['feature_dim']])
        test_z = tf.random.normal([batch_size, seq_len, self.config['z_dim']])
        
        # Test embedder
        _ = self.embedder(test_input)
        # Test generator
        _ = self.generator(test_z)
        # Test discriminator
        test_h = tf.random.normal([batch_size, seq_len, self.config['hidden_dim']])
        _ = self.discriminator(test_h)
        
        print(f"Models built successfully")
        print(f"  Embedder params: {self.embedder.count_params():,}")
        print(f"  Generator params: {self.generator.count_params():,}")
        print(f"  Discriminator params: {self.discriminator.count_params():,}")
    
    def compile(self, lr_g=1e-4, lr_d=5e-5):
        """Compile with optimizers"""
        self.opt_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.5)
        self.opt_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.5)
        self.opt_e = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.opt_s = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        
    def compute_air_quality_metrics(self, real, fake):
        """Compute air-quality specific metrics"""
        # Mean absolute error for pollution levels
        mae = tf.reduce_mean(tf.abs(real - fake))
        
        # Peak concentration matching
        real_peaks = tf.reduce_max(real, axis=1)
        fake_peaks = tf.reduce_max(fake, axis=1)
        peak_error = tf.reduce_mean(tf.abs(real_peaks - fake_peaks))
        
        # Diurnal pattern consistency
        real_diurnal = tf.reduce_mean(real, axis=[0, 2])  # Average over batch and features
        fake_diurnal = tf.reduce_mean(fake, axis=[0, 2])
        diurnal_error = tf.reduce_mean(tf.abs(real_diurnal - fake_diurnal))
        
        # Auto-correlation preservation
        def compute_autocorr(x, lag=3):
            x_centered = x - tf.reduce_mean(x, axis=1, keepdims=True)
            numerator = tf.reduce_mean(x_centered[:, :-lag] * x_centered[:, lag:], axis=1)
            denominator = tf.math.reduce_std(x_centered[:, :-lag], axis=1) * \
                         tf.math.reduce_std(x_centered[:, lag:], axis=1)
            return numerator / (denominator + 1e-8)
        
        autocorr_error = 0
        n_features = min(5, real.shape[2])
        for feat in range(n_features):
            real_ac = compute_autocorr(real[:, :, feat])
            fake_ac = compute_autocorr(fake[:, :, feat])
            autocorr_error += tf.reduce_mean(tf.abs(real_ac - fake_ac))
        autocorr_error /= n_features
        
        return {
            'mae': mae,
            'peak_error': peak_error,
            'diurnal_error': diurnal_error,
            'autocorr_error': autocorr_error
        }
    
    @tf.function
    def train_step(self, x_real):
        """Training step for air quality TimeGAN"""
        batch_size = tf.shape(x_real)[0]
        
        # Generate random noise
        z = tf.random.normal([batch_size, self.config['seq_len'], self.config['z_dim']])
        
        # ===== DISCRIMINATOR TRAINING =====
        d_losses = []
        for _ in range(self.config.get('n_critic', 1)):
            with tf.GradientTape() as tape:
                # Generate synthetic data
                e_fake = self.generator(z, training=True)
                h_fake = self.supervisor(e_fake, training=True)
                x_fake = self.recovery(h_fake, training=True)
                
                # Embed real data
                h_real = self.embedder(x_real, training=True)
                
                # Discriminator predictions
                d_real = self.discriminator(h_real, training=True)
                d_fake = self.discriminator(h_fake, training=True)
                
                # Discriminator loss
                d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
                
                # Gradient penalty
                epsilon = tf.random.uniform([batch_size, 1, 1])
                interpolated = epsilon * h_real + (1 - epsilon) * h_fake
                
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    d_interpolated = self.discriminator(interpolated, training=True)
                
                gradients = gp_tape.gradient(d_interpolated, interpolated)
                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-8)
                gp = tf.reduce_mean(tf.square(grad_norm - 1.0))
                
                total_d_loss = d_loss + self.config.get('lambda_gp', 10.0) * gp
            
            # Apply gradients
            d_grads = tape.gradient(total_d_loss, self.discriminator.trainable_variables)
            if d_grads is not None:
                d_grads = [tf.clip_by_norm(g, 1.0) for g in d_grads if g is not None]
                self.opt_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            
            d_losses.append(d_loss)
        
        # ===== GENERATOR TRAINING =====
        with tf.GradientTape() as tape:
            # Generate synthetic
            e_fake = self.generator(z, training=True)
            h_fake = self.supervisor(e_fake, training=True)
            x_fake = self.recovery(h_fake, training=True)
            
            # Embed real for reconstruction
            h_real = self.embedder(x_real, training=True)
            x_recon = self.recovery(h_real, training=True)
            
            # Supervisor loss
            h_hat = self.supervisor(h_real, training=True)
            supervisor_loss = tf.reduce_mean(tf.abs(h_real[:, 1:] - h_hat[:, :-1]))
            
            # Adversarial loss
            d_fake = self.discriminator(h_fake, training=False)
            g_adv_loss = -tf.reduce_mean(d_fake)
            
            # Reconstruction loss
            recon_loss = tf.reduce_mean(tf.abs(x_real - x_recon))
            
            # Air quality specific metrics
            air_metrics = self.compute_air_quality_metrics(x_real, x_fake)
            
            # Total generator loss
            total_g_loss = (
                g_adv_loss +
                self.config.get('lambda_sup', 0.2) * supervisor_loss +
                self.config.get('lambda_rec', 1.0) * recon_loss +
                self.config.get('lambda_air', 0.5) * air_metrics['mae']
            )
        
        # Apply gradients
        g_vars = (self.generator.trainable_variables + 
                 self.supervisor.trainable_variables +
                 self.recovery.trainable_variables)
        g_grads = tape.gradient(total_g_loss, g_vars)
        if g_grads is not None:
            g_grads = [tf.clip_by_norm(g, 1.0) for g in g_grads if g is not None]
            self.opt_g.apply_gradients(zip(g_grads, g_vars))
        
        # ===== EMBEDDER TRAINING =====
        with tf.GradientTape() as tape:
            h_real = self.embedder(x_real, training=True)
            x_recon = self.recovery(h_real, training=True)
            recon_loss = tf.reduce_mean(tf.abs(x_real - x_recon))
        
        e_grads = tape.gradient(recon_loss, self.embedder.trainable_variables)
        if e_grads is not None:
            e_grads = [tf.clip_by_norm(g, 1.0) for g in e_grads if g is not None]
            self.opt_e.apply_gradients(zip(e_grads, self.embedder.trainable_variables))
        
        return {
            'd_loss': tf.reduce_mean(d_losses) if d_losses else 0.0,
            'g_loss': total_g_loss,
            'recon_loss': recon_loss,
            'supervisor_loss': supervisor_loss,
            **air_metrics
        }
    
    def generate(self, n_samples, z_std=1.0):
        """Generate synthetic air quality data"""
        z = tf.random.normal([n_samples, self.config['seq_len'], 
                             self.config['z_dim']], stddev=z_std)
        
        e_fake = self.generator(z, training=False)
        h_fake = self.supervisor(e_fake, training=False)
        x_fake = self.recovery(h_fake, training=False)
        
        return x_fake.numpy()
    
    def save_weights(self, path):
        """Save model weights"""
        os.makedirs(path, exist_ok=True)
        self.generator.save_weights(os.path.join(path, 'generator.h5'))
        self.discriminator.save_weights(os.path.join(path, 'discriminator.h5'))
        self.embedder.save_weights(os.path.join(path, 'embedder.h5'))
        self.supervisor.save_weights(os.path.join(path, 'supervisor.h5'))
        self.recovery.save_weights(os.path.join(path, 'recovery.h5'))
        
    def load_weights(self, path):
        """Load model weights"""
        self.generator.load_weights(os.path.join(path, 'generator.h5'))
        self.discriminator.load_weights(os.path.join(path, 'discriminator.h5'))
        self.embedder.load_weights(os.path.join(path, 'embedder.h5'))
        self.supervisor.load_weights(os.path.join(path, 'supervisor.h5'))
        self.recovery.load_weights(os.path.join(path, 'recovery.h5'))