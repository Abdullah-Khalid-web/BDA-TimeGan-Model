# ============================================================================
# COMPLETE TIMEGAN IMPLEMENTATION FOR FINANCIAL TIME SERIES
# ============================================================================

import os
import sys
import time
import json
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. ENHANCED DATA PREPROCESSING
# ============================================================================

class BitcoinDataProcessor:
    def __init__(self, seq_len: int = 168, max_windows: int = 50000):
        self.seq_len = seq_len
        self.max_windows = max_windows
        self.scalers = {}
        self.feature_names = []
        
    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess Bitcoin data"""
        print(f"Loading data from {filepath}...")
        
        # Load data
        df = pd.read_csv(filepath, sep='\t', engine='python')
        print(f"Loaded {len(df)} rows")
        
        # Convert timestamp
        df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
        df = df.drop(columns=['Timestamp'])
        df = df.set_index('Datetime').sort_index()
        
        # Basic cleaning
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='time', limit_direction='both')
        df = df.ffill().bfill()
        
        # Remove periods with no trading
        if 'Volume' in df.columns:
            volume_ma = df['Volume'].rolling(60).mean()
            valid_mask = volume_ma > volume_ma.quantile(0.05)
            df = df[valid_mask]
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for financial time series"""
        enhanced_df = df.copy()
        
        # 1. Price-based features
        if 'Close' in enhanced_df.columns:
            # Returns
            enhanced_df['Returns'] = enhanced_df['Close'].pct_change()
            enhanced_df['Log_Returns'] = np.log(enhanced_df['Close'] / enhanced_df['Close'].shift(1))
            
            # Price range
            if all(col in enhanced_df.columns for col in ['High', 'Low']):
                enhanced_df['Price_Range'] = (enhanced_df['High'] - enhanced_df['Low']) / enhanced_df['Close'].shift(1)
            
            # Moving averages
            enhanced_df['MA_7'] = enhanced_df['Close'].rolling(7).mean()
            enhanced_df['MA_21'] = enhanced_df['Close'].rolling(21).mean()
            enhanced_df['MA_50'] = enhanced_df['Close'].rolling(50).mean()
            
            # Exponential moving averages
            enhanced_df['EMA_12'] = enhanced_df['Close'].ewm(span=12).mean()
            enhanced_df['EMA_26'] = enhanced_df['Close'].ewm(span=26).mean()
            
            # MACD
            enhanced_df['MACD'] = enhanced_df['EMA_12'] - enhanced_df['EMA_26']
            enhanced_df['MACD_Signal'] = enhanced_df['MACD'].ewm(span=9).mean()
            enhanced_df['MACD_Hist'] = enhanced_df['MACD'] - enhanced_df['MACD_Signal']
            
            # RSI
            delta = enhanced_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            enhanced_df['RSI'] = 100 - (100 / (1 + rs))
            
            # Volatility measures
            enhanced_df['Volatility_5'] = enhanced_df['Returns'].rolling(5).std()
            enhanced_df['Volatility_20'] = enhanced_df['Returns'].rolling(20).std()
            
            # Momentum
            enhanced_df['Momentum_5'] = enhanced_df['Close'].pct_change(5)
            enhanced_df['Momentum_10'] = enhanced_df['Close'].pct_change(10)
        
        # 2. Volume features
        if 'Volume' in enhanced_df.columns:
            enhanced_df['Volume_MA_5'] = enhanced_df['Volume'].rolling(5).mean()
            enhanced_df['Volume_MA_20'] = enhanced_df['Volume'].rolling(20).mean()
            enhanced_df['Volume_Ratio'] = enhanced_df['Volume'] / enhanced_df['Volume_MA_20']
            enhanced_df['Volume_Spike'] = (enhanced_df['Volume'] / enhanced_df['Volume_MA_5']).clip(upper=10)
        
        # 3. Cyclical time features
        enhanced_df['Hour_sin'] = np.sin(2 * np.pi * enhanced_df.index.hour / 24)
        enhanced_df['Hour_cos'] = np.cos(2 * np.pi * enhanced_df.index.hour / 24)
        enhanced_df['Day_sin'] = np.sin(2 * np.pi * enhanced_df.index.dayofweek / 7)
        enhanced_df['Day_cos'] = np.cos(2 * np.pi * enhanced_df.index.dayofweek / 7)
        
        # Drop NaN values
        enhanced_df = enhanced_df.dropna()
        
        # Remove extreme outliers
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos']
        for col in numeric_cols:
            if col not in exclude_cols:
                Q1 = enhanced_df[col].quantile(0.01)
                Q3 = enhanced_df[col].quantile(0.99)
                enhanced_df[col] = enhanced_df[col].clip(Q1, Q3)
        
        self.feature_names = enhanced_df.columns.tolist()
        print(f"Created {len(self.feature_names)} features")
        
        return enhanced_df
    
    def create_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """Create sequences from dataframe"""
        data = df.values.astype(np.float32)
        n_samples, n_features = data.shape
        
        # Calculate number of possible sequences
        max_start = n_samples - self.seq_len
        if max_start <= 0:
            raise ValueError(f"Data too short for sequence length {self.seq_len}")
        
        # Create overlapping sequences
        all_starts = np.arange(0, max_start, 1)  # Stride of 1 for maximum overlap
        
        # Randomly sample sequences
        if len(all_starts) > self.max_windows:
            selected_starts = np.random.choice(all_starts, self.max_windows, replace=False)
        else:
            selected_starts = all_starts
        
        selected_starts = np.sort(selected_starts)
        
        # Create sequences array
        sequences = np.zeros((len(selected_starts), self.seq_len, n_features), dtype=np.float32)
        
        for i, start in enumerate(selected_starts):
            sequences[i] = data[start:start + self.seq_len]
        
        return sequences
    
    def scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Scale sequences using StandardScaler"""
        n_sequences, seq_len, n_features = sequences.shape
        
        # Reshape for scaling
        sequences_2d = sequences.reshape(-1, n_features)
        
        # Fit scaler
        scaler = StandardScaler()
        scaled_2d = scaler.fit_transform(sequences_2d)
        
        # Save scaler for inverse transformation
        self.scalers['standard'] = scaler
        
        # Reshape back to 3D
        scaled_sequences = scaled_2d.reshape(n_sequences, seq_len, n_features)
        
        return scaled_sequences
    
    def process(self, input_path: str, output_dir: str) -> Dict:
        """Complete processing pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load and preprocess
        df = self.load_and_preprocess(input_path)
        
        # 2. Create features
        df = self.create_features(df)
        
        # 3. Create sequences
        sequences = self.create_sequences(df)
        print(f"Created {sequences.shape[0]} sequences")
        
        # 4. Scale sequences
        scaled_sequences = self.scale_sequences(sequences)
        
        # 5. Split into train/val/test
        np.random.shuffle(scaled_sequences)
        n_total = len(scaled_sequences)
        
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val
        
        train_data = scaled_sequences[:n_train]
        val_data = scaled_sequences[n_train:n_train + n_val]
        test_data = scaled_sequences[n_train + n_val:]
        
        # 6. Save data
        np.save(os.path.join(output_dir, 'train.npy'), train_data)
        np.save(os.path.join(output_dir, 'val.npy'), val_data)
        np.save(os.path.join(output_dir, 'test.npy'), test_data)
        
        # 7. Save metadata
        with open(os.path.join(output_dir, 'features.txt'), 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        meta = {
            'total_sequences': n_total,
            'train_size': n_train,
            'val_size': n_val,
            'test_size': n_test,
            'sequence_length': self.seq_len,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Train: {n_train} sequences")
        print(f"   Val: {n_val} sequences")
        print(f"   Test: {n_test} sequences")
        
        return meta

# ============================================================================
# 2. ENHANCED TIMEGAN MODEL ARCHITECTURE
# ============================================================================

class EnhancedEmbedder(Model):
    """Enhanced embedder with bidirectional LSTM and attention"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.Bidirectional(
                layers.LSTM(hidden_dim // 2, return_sequences=True),
                name=f'embedder_bilstm_{i}'
            )
            self.lstm_layers.append(lstm)
        
        # Layer normalization
        self.layer_norm = layers.LayerNormalization()
        
        # Output projection
        self.dense = layers.Dense(hidden_dim, activation='tanh')
    
    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        x = self.layer_norm(x)
        return self.dense(x)

class EnhancedGenerator(Model):
    """Enhanced generator with temporal convolutions"""
    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.dense_in = layers.Dense(hidden_dim, activation='relu')
        
        # Temporal convolutional layers
        self.conv_layers = []
        for i in range(2):
            conv = layers.Conv1D(
                hidden_dim, kernel_size=3, padding='same',
                activation='relu', kernel_regularizer=l2(1e-5)
            )
            self.conv_layers.append(conv)
        
        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_dim, return_sequences=True)
            self.lstm_layers.append(lstm)
        
        # Output layer
        self.dense_out = layers.Dense(hidden_dim, activation='tanh')
        self.layer_norm = layers.LayerNormalization()
    
    def call(self, z, training=False):
        x = self.dense_in(z)
        
        # Convolutional processing
        for conv in self.conv_layers:
            x = conv(x, training=training)
        
        # LSTM processing
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        
        x = self.dense_out(x)
        return self.layer_norm(x)

class EnhancedDiscriminator(Model):
    """Enhanced discriminator with convolutional layers"""
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = []
        filters = [64, 128]
        for i, filters_num in enumerate(filters):
            conv = layers.Conv1D(
                filters_num, kernel_size=3, strides=2,
                padding='same', activation='leaky_relu'
            )
            self.conv_layers.append(conv)
        
        # LSTM layer
        self.lstm = layers.LSTM(128, return_sequences=True)
        
        # Output layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='leaky_relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(128, activation='leaky_relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        for conv in self.conv_layers:
            x = conv(x, training=training)
        
        x = self.lstm(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)

class Supervisor(Model):
    """Supervisor network"""
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_dim, return_sequences=True)
            self.lstm_layers.append(lstm)
        self.dense = layers.Dense(hidden_dim, activation='tanh')
    
    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        return self.dense(x)

class Recovery(Model):
    """Recovery network"""
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_dim, return_sequences=True)
            self.lstm_layers.append(lstm)
        self.dense = layers.Dense(output_dim, activation='linear')
    
    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        return self.dense(x)

# ============================================================================
# 3. TIMEGAN TRAINER WITH CURRICULUM LEARNING
# ============================================================================

class TimeGANTrainer:
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize models
        self.embedder = EnhancedEmbedder(
            config['feature_dim'], 
            config['hidden_dim'],
            num_layers=2
        )
        self.generator = EnhancedGenerator(
            config['z_dim'],
            config['hidden_dim'],
            num_layers=2
        )
        self.discriminator = EnhancedDiscriminator(
            config['hidden_dim'],
            num_layers=1
        )
        self.supervisor = Supervisor(
            config['hidden_dim'],
            num_layers=1
        )
        self.recovery = Recovery(
            config['hidden_dim'],
            config['feature_dim'],
            num_layers=2
        )
        
        # Initialize optimizers
        self.opt_g = Adam(config['lr_g'], beta_1=0.5)
        self.opt_d = Adam(config['lr_d'], beta_1=0.5)
        self.opt_s = Adam(config['lr_s'], beta_1=0.5)
        self.opt_e = Adam(config['lr_e'], beta_1=0.5)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience = 0
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
    def sample_z(self, batch_size: int) -> tf.Tensor:
        """Sample random noise"""
        return tf.random.normal([batch_size, self.config['seq_len'], self.config['z_dim']])
    
    def gradient_penalty(self, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha * real + (1.0 - alpha) * fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            d_interpolated = self.discriminator(interpolated, training=True)
        
        gradients = tape.gradient(d_interpolated, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]) + 1e-8)
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        return gp
    
    @tf.function
    def train_step_supervised(self, x_real: tf.Tensor):
        """Supervised training step"""
        with tf.GradientTape() as tape:
            # Embed real data
            h_real = self.embedder(x_real, training=True)
            
            # Supervised prediction
            h_hat = self.supervisor(h_real, training=True)
            
            # Supervised loss
            loss = tf.reduce_mean(tf.keras.losses.MSE(h_real[:, 1:], h_hat[:, :-1]))
        
        # Compute gradients
        grads = tape.gradient(loss, self.supervisor.trainable_variables)
        
        # Apply gradients
        self.opt_s.apply_gradients(zip(grads, self.supervisor.trainable_variables))
        
        return loss
    
    @tf.function
    def train_step_critic(self, x_real: tf.Tensor):
        """Critic (discriminator) training step"""
        batch_size = tf.shape(x_real)[0]
        z = self.sample_z(batch_size)
        
        with tf.GradientTape() as tape:
            # Generate synthetic data
            e_fake = self.generator(z, training=True)
            h_fake = self.supervisor(e_fake, training=True)
            
            # Embed real data
            h_real = self.embedder(x_real, training=True)
            
            # Discriminator outputs
            d_real = self.discriminator(h_real, training=True)
            d_fake = self.discriminator(h_fake, training=True)
            
            # WGAN loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            
            # Gradient penalty
            gp = self.gradient_penalty(h_real, h_fake)
            
            # Total loss
            total_loss = d_loss + self.config['lambda_gp'] * gp
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.opt_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        return d_loss, gp
    
    @tf.function
    def train_step_generator(self, x_real: tf.Tensor):
        """Generator training step"""
        batch_size = tf.shape(x_real)[0]
        z = self.sample_z(batch_size)
        
        with tf.GradientTape() as tape:
            # Generate synthetic data
            e_fake = self.generator(z, training=True)
            h_fake = self.supervisor(e_fake, training=True)
            x_fake = self.recovery(h_fake, training=True)
            
            # Discriminator output
            d_fake = self.discriminator(h_fake, training=False)
            
            # Embed real data
            h_real = self.embedder(x_real, training=True)
            x_recon = self.recovery(h_real, training=True)
            
            # Loss components
            # 1. Adversarial loss
            adv_loss = -tf.reduce_mean(d_fake)
            
            # 2. Supervised loss
            h_hat = self.supervisor(h_real, training=False)
            sup_loss = tf.reduce_mean(tf.keras.losses.MSE(h_real[:, 1:], h_hat[:, :-1]))
            
            # 3. Reconstruction loss
            recon_loss = tf.reduce_mean(tf.keras.losses.MSE(x_real, x_recon))
            
            # 4. Statistical matching loss
            real_mean = tf.reduce_mean(x_real, axis=[0, 1])
            fake_mean = tf.reduce_mean(x_fake, axis=[0, 1])
            stat_loss = tf.reduce_mean(tf.square(real_mean - fake_mean))
            
            # Total loss
            total_loss = (
                adv_loss +
                self.config['lambda_sup'] * sup_loss +
                self.config['lambda_rec'] * recon_loss +
                self.config['lambda_stat'] * stat_loss
            )
        
        # Variables to train
        trainable_vars = (
            self.generator.trainable_variables +
            self.supervisor.trainable_variables +
            self.recovery.trainable_variables +
            self.embedder.trainable_variables
        )
        
        # Compute gradients
        grads = tape.gradient(total_loss, trainable_vars)
        
        # Clip gradients
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        
        # Apply gradients
        self.opt_g.apply_gradients(zip(grads, trainable_vars))
        
        return {
            'total': total_loss,
            'adv': adv_loss,
            'sup': sup_loss,
            'recon': recon_loss,
            'stat': stat_loss
        }
    
    def validate(self, val_data: np.ndarray) -> Dict:
        """Validate model performance"""
        # Sample validation data
        n_samples = min(100, len(val_data))
        val_batch = val_data[:n_samples]
        
        # Generate synthetic data
        z = self.sample_z(n_samples)
        e_fake = self.generator(z, training=False)
        h_fake = self.supervisor(e_fake, training=False)
        x_fake = self.recovery(h_fake, training=False).numpy()
        
        # Compute metrics
        metrics = {}
        
        # 1. Mean absolute difference
        real_mean = np.mean(val_batch, axis=(0, 1))
        fake_mean = np.mean(x_fake, axis=(0, 1))
        metrics['mean_diff'] = float(np.mean(np.abs(real_mean - fake_mean)))
        
        # 2. Standard deviation difference
        real_std = np.std(val_batch, axis=(0, 1))
        fake_std = np.std(x_fake, axis=(0, 1))
        metrics['std_diff'] = float(np.mean(np.abs(real_std - fake_std)))
        
        # 3. Reconstruction error on real data
        h_real = self.embedder(tf.convert_to_tensor(val_batch), training=False)
        x_recon = self.recovery(h_real, training=False).numpy()
        metrics['recon_error'] = float(np.mean(np.abs(val_batch - x_recon)))
        
        return metrics
    
    def train(self, train_data: np.ndarray, val_data: np.ndarray):
        """Main training loop"""
        print("Starting TimeGAN training...")
        
        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(10000).batch(self.config['batch_size']).prefetch(2)
        
        # Phase 1: Supervised pretraining (20 epochs)
        print("\n=== Phase 1: Supervised Pretraining ===")
        for epoch in range(self.config['supervised_epochs']):
            losses = []
            for batch in train_dataset:
                loss = self.train_step_supervised(batch)
                losses.append(loss.numpy())
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self.config['supervised_epochs']}: "
                      f"Loss = {np.mean(losses):.6f}")
        
        # Phase 2: Joint training
        print("\n=== Phase 2: Joint Training ===")
        for epoch in range(self.config['adversarial_epochs']):
            start_time = time.time()
            
            d_losses, g_losses = [], []
            d_loss_vals, gp_vals = [], []
            
            for batch in train_dataset:
                # Train discriminator multiple times
                for _ in range(self.config['n_critic']):
                    d_loss, gp = self.train_step_critic(batch)
                    d_losses.append(d_loss.numpy())
                    gp_vals.append(gp.numpy())
                
                # Train generator once
                g_loss_dict = self.train_step_generator(batch)
                g_losses.append(g_loss_dict['total'].numpy())
                d_loss_vals.append(g_loss_dict['adv'].numpy())
            
            # Validation
            if (epoch + 1) % self.config['val_frequency'] == 0:
                val_metrics = self.validate(val_data)
                val_score = val_metrics['mean_diff']
                
                print(f"\nEpoch {epoch + 1}/{self.config['adversarial_epochs']}:")
                print(f"  D_loss: {np.mean(d_losses):.4f}, G_loss: {np.mean(g_losses):.4f}")
                print(f"  Val mean diff: {val_metrics['mean_diff']:.6f}")
                print(f"  Val std diff: {val_metrics['std_diff']:.6f}")
                print(f"  Recon error: {val_metrics['recon_error']:.6f}")
                
                # Save best model
                if val_score < self.best_val_loss:
                    self.best_val_loss = val_score
                    self.patience = 0
                    self.save_checkpoint(epoch + 1, val_score)
                    print(f"  ✓ New best model saved!")
                else:
                    self.patience += 1
                
                # Early stopping
                if self.patience >= self.config['patience']:
                    print(f"\n⚠️ Early stopping triggered!")
                    break
            
            epoch_time = time.time() - start_time
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Time = {epoch_time:.2f}s")
        
        print("\n✅ Training completed!")
    
    def save_checkpoint(self, epoch: int, val_score: float):
        """Save model checkpoint"""
        checkpoint_dir = self.config['checkpoint_dir']
        
        # Save weights
        self.generator.save_weights(
            os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.h5')
        )
        self.discriminator.save_weights(
            os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.h5')
        )
        self.supervisor.save_weights(
            os.path.join(checkpoint_dir, f'supervisor_epoch_{epoch}.h5')
        )
        self.recovery.save_weights(
            os.path.join(checkpoint_dir, f'recovery_epoch_{epoch}.h5')
        )
        self.embedder.save_weights(
            os.path.join(checkpoint_dir, f'embedder_epoch_{epoch}.h5')
        )
        
        # Save training state
        state = {
            'epoch': epoch,
            'best_val_loss': float(self.best_val_loss),
            'patience': self.patience,
            'val_score': float(val_score)
        }
        
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples"""
        print(f"Generating {n_samples} synthetic samples...")
        
        synthetic_batches = []
        batch_size = self.config['batch_size']
        
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            
            # Sample noise
            z = self.sample_z(current_batch)
            
            # Generate
            e_fake = self.generator(z, training=False)
            h_fake = self.supervisor(e_fake, training=False)
            x_fake = self.recovery(h_fake, training=False).numpy()
            
            synthetic_batches.append(x_fake)
        
        synthetic = np.concatenate(synthetic_batches, axis=0)
        
        print(f"✅ Generated {synthetic.shape[0]} samples")
        return synthetic

# ============================================================================
# 4. EVALUATION AND VISUALIZATION
# ============================================================================

class TimeGANEvaluator:
    def __init__(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        self.real = real_data
        self.synth = synthetic_data
    
    def compute_metrics(self) -> Dict:
        """Compute comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic statistics
        real_flat = self.real.reshape(-1, self.real.shape[2])
        synth_flat = self.synth.reshape(-1, self.synth.shape[2])
        
        # 1. Mean and std similarity
        real_mean = np.mean(real_flat, axis=0)
        synth_mean = np.mean(synth_flat, axis=0)
        mean_diff = np.mean(np.abs(real_mean - synth_mean))
        
        real_std = np.std(real_flat, axis=0)
        synth_std = np.std(synth_flat, axis=0)
        std_diff = np.mean(np.abs(real_std - synth_std))
        
        metrics['mean_difference'] = float(mean_diff)
        metrics['std_difference'] = float(std_diff)
        
        # 2. Distribution similarity (using simple correlation)
        corr_matrix = np.corrcoef(real_flat.T, synth_flat.T)
        n_features = real_flat.shape[1]
        similarity = np.mean(np.diag(corr_matrix[:n_features, n_features:]))
        metrics['distribution_similarity'] = float(similarity)
        
        # 3. Temporal dynamics (autocorrelation)
        def compute_avg_autocorr(data, max_lag=10):
            autocorrs = []
            for i in range(data.shape[0]):
                for f in range(min(3, data.shape[2])):  # First 3 features
                    series = data[i, :, f]
                    for lag in range(1, min(max_lag, len(series)-1)):
                        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        autocorrs.append(np.abs(corr))
            return np.mean(autocorrs) if autocorrs else 0
        
        real_autocorr = compute_avg_autocorr(self.real[:100])  # Sample 100 sequences
        synth_autocorr = compute_avg_autocorr(self.synth[:100])
        metrics['autocorr_difference'] = float(np.abs(real_autocorr - synth_autocorr))
        
        # 4. PCA similarity
        pca = PCA(n_components=2)
        combined = np.vstack([real_flat[:1000], synth_flat[:1000]])
        pca.fit(combined)
        
        real_pca = pca.transform(real_flat[:1000])
        synth_pca = pca.transform(synth_flat[:1000])
        
        pca_mean_diff = np.mean(np.abs(np.mean(real_pca, axis=0) - np.mean(synth_pca, axis=0)))
        metrics['pca_similarity'] = float(1.0 / (1.0 + pca_mean_diff))
        
        # Overall score (weighted average)
        weights = {
            'mean_difference': 0.3,
            'std_difference': 0.3,
            'distribution_similarity': 0.2,
            'autocorr_difference': 0.1,
            'pca_similarity': 0.1
        }
        
        # Convert differences to similarities
        mean_sim = 1.0 / (1.0 + metrics['mean_difference'])
        std_sim = 1.0 / (1.0 + metrics['std_difference'])
        autocorr_sim = 1.0 / (1.0 + metrics['autocorr_difference'])
        
        overall_score = (
            weights['mean_difference'] * mean_sim +
            weights['std_difference'] * std_sim +
            weights['distribution_similarity'] * metrics['distribution_similarity'] +
            weights['autocorr_difference'] * autocorr_sim +
            weights['pca_similarity'] * metrics['pca_similarity']
        )
        
        metrics['overall_score'] = float(overall_score)
        metrics['overall_percentage'] = float(overall_score * 100)
        
        return metrics
    
    def visualize_comparison(self, save_path: str = None):
        """Visualize real vs synthetic data comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Sample sequences
        real_sample = self.real[:5]
        synth_sample = self.synth[:5]
        
        # 1. Price comparison (first feature)
        ax = axes[0, 0]
        for i in range(min(5, len(real_sample))):
            ax.plot(real_sample[i, :, 0], alpha=0.6, label=f'Real {i+1}' if i == 0 else None)
            ax.plot(synth_sample[i, :, 0], alpha=0.6, linestyle='--', 
                   label=f'Synth {i+1}' if i == 0 else None)
        ax.set_title('Price Sequences')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (scaled)')
        ax.legend()
        
        # 2. Distribution comparison
        ax = axes[0, 1]
        real_flat = self.real[:, :, 0].flatten()
        synth_flat = self.synth[:, :, 0].flatten()
        
        # Sample for histogram
        real_sample_hist = np.random.choice(real_flat, min(10000, len(real_flat)))
        synth_sample_hist = np.random.choice(synth_flat, min(10000, len(synth_flat)))
        
        ax.hist(real_sample_hist, bins=50, alpha=0.5, density=True, label='Real')
        ax.hist(synth_sample_hist, bins=50, alpha=0.5, density=True, label='Synthetic')
        ax.set_title('Price Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
        # 3. Volatility comparison
        ax = axes[0, 2]
        # Compute returns
        real_returns = np.diff(self.real[:100, :, 0], axis=1)
        synth_returns = np.diff(self.synth[:100, :, 0], axis=1)
        
        real_vol = np.std(real_returns, axis=1)
        synth_vol = np.std(synth_returns, axis=1)
        
        ax.boxplot([real_vol, synth_vol], labels=['Real', 'Synthetic'])
        ax.set_title('Volatility Distribution')
        ax.set_ylabel('Volatility')
        
        # 4. Autocorrelation comparison
        ax = axes[1, 0]
        max_lag = 20
        
        def compute_autocorr(series, max_lag):
            autocorrs = []
            for lag in range(1, min(max_lag, len(series)-1)):
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                autocorrs.append(corr)
            return autocorrs
        
        # Sample autocorrelations
        real_ac = []
        synth_ac = []
        for i in range(min(50, len(self.real))):
            real_series = self.real[i, :, 0]
            synth_series = self.synth[i, :, 0]
            real_ac.append(compute_autocorr(real_series, max_lag))
            synth_ac.append(compute_autocorr(synth_series, max_lag))
        
        real_ac_mean = np.mean(real_ac, axis=0)
        synth_ac_mean = np.mean(synth_ac, axis=0)
        
        lags = np.arange(1, len(real_ac_mean) + 1)
        ax.plot(lags, real_ac_mean, label='Real', marker='o')
        ax.plot(lags, synth_ac_mean, label='Synthetic', marker='s')
        ax.set_title('Autocorrelation Comparison')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.legend()
        
        # 5. PCA visualization
        ax = axes[1, 1]
        pca = PCA(n_components=2)
        combined = np.vstack([
            self.real[:1000].reshape(-1, self.real.shape[2]),
            self.synth[:1000].reshape(-1, self.synth.shape[2])
        ])
        pca.fit(combined)
        
        real_pca = pca.transform(self.real[:1000].reshape(-1, self.real.shape[2]))
        synth_pca = pca.transform(self.synth[:1000].reshape(-1, self.synth.shape[2]))
        
        ax.scatter(real_pca[:500, 0], real_pca[:500, 1], alpha=0.5, label='Real', s=10)
        ax.scatter(synth_pca[:500, 0], synth_pca[:500, 1], alpha=0.5, label='Synthetic', s=10)
        ax.set_title('PCA Visualization')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        
        # 6. Metrics summary
        ax = axes[1, 2]
        metrics = self.compute_metrics()
        
        metric_names = ['Mean Diff', 'Std Diff', 'Dist Sim', 'Autocorr Diff', 'PCA Sim']
        metric_values = [
            metrics['mean_difference'],
            metrics['std_difference'],
            metrics['distribution_similarity'],
            metrics['autocorr_difference'],
            metrics['pca_similarity']
        ]
        
        colors = ['red' if 'Diff' in name else 'green' for name in metric_names]
        
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_title('Quality Metrics')
        ax.set_ylabel('Value')
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add overall score
        ax.text(0.5, 0.95, f'Overall Score: {metrics["overall_percentage"]:.1f}%',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
        return metrics

# ============================================================================
# 5. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TimeGAN for Financial Time Series')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['preprocess', 'train', 'generate', 'evaluate'],
                       help='Mode to run: preprocess, train, generate, or evaluate')
    parser.add_argument('--input', type=str, help='Input data path')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--n_samples', type=int, default=50000,
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'preprocess':
        # Preprocess data
        processor = BitcoinDataProcessor(seq_len=168, max_windows=50000)
        meta = processor.process(args.input, args.output_dir)
        
        print(f"\n📊 Preprocessing Summary:")
        print(f"   Total sequences: {meta['total_sequences']}")
        print(f"   Sequence length: {meta['sequence_length']}")
        print(f"   Number of features: {meta['num_features']}")
        
        # Save configuration
        config = {
            'seq_len': meta['sequence_length'],
            'feature_dim': meta['num_features'],
            'hidden_dim': 128,
            'z_dim': 64,
            'batch_size': 64,
            'supervised_epochs': 20,
            'adversarial_epochs': 200,
            'n_critic': 5,
            'val_frequency': 5,
            'patience': 20,
            'lr_g': 5e-5,
            'lr_d': 1e-5,
            'lr_s': 1e-4,
            'lr_e': 1e-4,
            'lambda_gp': 10.0,
            'lambda_sup': 0.2,
            'lambda_rec': 1.0,
            'lambda_stat': 5.0,
            'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints')
        }
        
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {os.path.join(args.output_dir, 'config.json')}")
    
    elif args.mode == 'train':
        # Load configuration
        config_path = os.path.join(args.output_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            print("Please run preprocessing first.")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load data
        print("Loading training data...")
        train_data = np.load(os.path.join(args.output_dir, 'train.npy'))
        val_data = np.load(os.path.join(args.output_dir, 'val.npy'))
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Val data shape: {val_data.shape}")
        
        # Initialize and train model
        trainer = TimeGANTrainer(config)
        trainer.train(train_data, val_data)
        
        print(f"\n✅ Training completed! Checkpoints saved to {config['checkpoint_dir']}")
    
    elif args.mode == 'generate':
        # Load configuration
        config_path = os.path.join(args.output_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize trainer and load latest weights
        trainer = TimeGANTrainer(config)
        
        # Find latest checkpoint
        checkpoint_dir = config['checkpoint_dir']
        if os.path.exists(checkpoint_dir):
            # Look for generator weights
            weight_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('generator_epoch_')]
            if weight_files:
                # Get latest epoch
                epochs = [int(f.split('_')[2].split('.')[0]) for f in weight_files]
                latest_epoch = max(epochs)
                
                # Load weights
                print(f"Loading weights from epoch {latest_epoch}...")
                
                trainer.generator.load_weights(
                    os.path.join(checkpoint_dir, f'generator_epoch_{latest_epoch}.h5')
                )
                trainer.supervisor.load_weights(
                    os.path.join(checkpoint_dir, f'supervisor_epoch_{latest_epoch}.h5')
                )
                trainer.recovery.load_weights(
                    os.path.join(checkpoint_dir, f'recovery_epoch_{latest_epoch}.h5')
                )
                
                # Generate synthetic data
                synthetic_data = trainer.generate(args.n_samples)
                
                # Save synthetic data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(args.output_dir, f'synthetic_{timestamp}.npy')
                np.save(save_path, synthetic_data)
                
                print(f"✅ Generated {synthetic_data.shape[0]} synthetic sequences")
                print(f"✅ Saved to {save_path}")
            else:
                print("❌ No model weights found. Please train the model first.")
        else:
            print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
    
    elif args.mode == 'evaluate':
        # Load real data
        print("Loading data for evaluation...")
        
        real_data = []
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(args.output_dir, f'{split}.npy')
            if os.path.exists(split_path):
                data = np.load(split_path)
                real_data.append(data)
        
        if not real_data:
            print("❌ No real data found. Please preprocess data first.")
            return
        
        real_data = np.concatenate(real_data, axis=0)
        
        # Load synthetic data
        synthetic_files = [f for f in os.listdir(args.output_dir) if f.startswith('synthetic_')]
        if not synthetic_files:
            print("❌ No synthetic data found. Please generate synthetic data first.")
            return
        
        # Use latest synthetic data
        latest_synthetic = sorted(synthetic_files)[-1]
        synthetic_path = os.path.join(args.output_dir, latest_synthetic)
        synthetic_data = np.load(synthetic_path)
        
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        
        # Evaluate
        evaluator = TimeGANEvaluator(real_data[:1000], synthetic_data[:1000])
        
        # Compute metrics
        print("\n📊 Computing evaluation metrics...")
        metrics = evaluator.compute_metrics()
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key:25}: {value:.6f}")
        print("="*50)
        
        print(f"\n🎯 OVERALL QUALITY SCORE: {metrics['overall_percentage']:.1f}%")
        
        if metrics['overall_percentage'] >= 85:
            print("✅ Excellent quality! Synthetic data is highly realistic.")
        elif metrics['overall_percentage'] >= 70:
            print("👍 Good quality! Synthetic data is reasonably realistic.")
        elif metrics['overall_percentage'] >= 50:
            print("⚠️ Fair quality. Room for improvement.")
        else:
            print("❌ Poor quality. Consider retraining the model.")
        
        # Visualize
        print("\n📈 Generating visualization...")
        viz_path = os.path.join(args.output_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        evaluator.visualize_comparison(save_path=viz_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Metrics saved to {metrics_path}")

if __name__ == '__main__':
    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ GPU error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")
    
    # Run main
    main()