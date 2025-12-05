# train_timegan_curriculum.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
import json
import time
from timegan_enhanced_tf import (
    EnhancedEmbedder, EnhancedGenerator,
    EnhancedDiscriminator, EnhancedSupervisor, EnhancedRecovery
)

class CurriculumTimeGANTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_models()
        self.setup_optimizers()
        self.setup_training_state()
        
    def setup_models(self):
        # Initialize enhanced models
        self.embedder = EnhancedEmbedder(
            self.config['feature_dim'], 
            self.config['hidden_dim'],
            num_layers=3,
            dropout=0.1
        )
        self.generator = EnhancedGenerator(
            self.config['z_dim'],
            self.config['hidden_dim'],
            num_layers=3
        )
        self.discriminator = EnhancedDiscriminator(
            self.config['hidden_dim']
        )
        self.supervisor = EnhancedSupervisor(
            self.config['hidden_dim'],
            num_layers=2
        )
        self.recovery = EnhancedRecovery(
            self.config['hidden_dim'],
            self.config['feature_dim'],
            num_layers=2
        )
        
        # Build models
        self.embedder.build(input_shape=(None, self.config['seq_len'], 
                                       self.config['feature_dim']))
        self.generator.build(input_shape=(None, self.config['seq_len'], 
                                         self.config['z_dim']))
        
    def setup_optimizers(self):
        # Learning rate schedule
        self.lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000,
            decay_rate=0.96
        )
        self.lr_schedule_d = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=1000,
            decay_rate=0.96
        )
        
        self.opt_g = optimizers.Adam(self.lr_schedule_g, beta_1=0.5)
        self.opt_d = optimizers.Adam(self.lr_schedule_d, beta_1=0.5)
        self.opt_s = optimizers.Adam(1e-4, beta_1=0.5)
        self.opt_e = optimizers.Adam(1e-4, beta_1=0.5)
        
    def setup_training_state(self):
        self.phase = 1  # Curriculum phase
        self.best_fid = float('inf')
        self.patience = 0
        
    def compute_statistical_loss(self, real, fake):
        """Enhanced statistical matching loss"""
        # Mean and std matching
        real_mean = tf.reduce_mean(real, axis=[0, 1])
        fake_mean = tf.reduce_mean(fake, axis=[0, 1])
        mean_loss = tf.reduce_mean(tf.abs(real_mean - fake_mean))
        
        real_std = tf.math.reduce_std(real, axis=[0, 1])
        fake_std = tf.math.reduce_std(fake, axis=[0, 1])
        std_loss = tf.reduce_mean(tf.abs(real_std - fake_std))
        
        # Autocorrelation matching
        def autocorr(x, lag=5):
            x_centered = x - tf.reduce_mean(x, axis=1, keepdims=True)
            autocorrs = []
            for l in range(1, lag+1):
                corr = tf.reduce_mean(
                    x_centered[:, :-l] * x_centered[:, l:], axis=1
                ) / (tf.math.reduce_std(x_centered[:, :-l], axis=1) * 
                     tf.math.reduce_std(x_centered[:, l:], axis=1) + 1e-8)
                autocorrs.append(tf.reduce_mean(tf.abs(corr)))
            return tf.reduce_mean(autocorrs)
        
        autocorr_loss = 0
        for feat in range(real.shape[2]):
            real_ac = autocorr(real[:, :, feat])
            fake_ac = autocorr(fake[:, :, feat])
            autocorr_loss += tf.abs(real_ac - fake_ac)
        autocorr_loss /= real.shape[2]
        
        # Volatility clustering loss
        real_returns = real[:, 1:, 0] - real[:, :-1, 0]
        fake_returns = fake[:, 1:, 0] - fake[:, :-1, 0]
        
        real_vol_clust = tf.reduce_mean(tf.abs(real_returns[:, 1:] * real_returns[:, :-1]))
        fake_vol_clust = tf.reduce_mean(tf.abs(fake_returns[:, 1:] * fake_returns[:, :-1]))
        vol_clust_loss = tf.abs(real_vol_clust - fake_vol_clust)
        
        return (mean_loss + std_loss + autocorr_loss + vol_clust_loss) / 4
    
    @tf.function
    def train_step_phase1(self, x_real):
        """Phase 1: Focus on reconstruction and basic patterns"""
        batch_size = tf.shape(x_real)[0]
        z = tf.random.normal([batch_size, self.config['seq_len'], 
                             self.config['z_dim']])
        
        with tf.GradientTape() as tape:
            # Generate synthetic
            e_fake = self.generator(z, training=True)
            h_fake = self.supervisor(e_fake, training=True)
            x_fake = self.recovery(h_fake, training=True)
            
            # Embed real
            h_real = self.embedder(x_real, training=True)
            x_recon = self.recovery(h_real, training=True)
            
            # Reconstruction loss (weighted higher in phase 1)
            recon_loss = tf.reduce_mean(tf.abs(x_real - x_recon))
            
            # Basic statistical matching
            stat_loss = self.compute_statistical_loss(x_real, x_fake)
            
            # Supervised loss
            h_hat = self.supervisor(h_real, training=True)
            sup_loss = tf.reduce_mean(tf.abs(h_real[:, 1:] - h_hat[:, :-1]))
            
            total_loss = (recon_loss * 2.0 + 
                         stat_loss * 1.0 + 
                         sup_loss * 0.5)
        
        vars_to_train = (self.generator.trainable_variables + 
                        self.supervisor.trainable_variables +
                        self.recovery.trainable_variables +
                        self.embedder.trainable_variables)
        
        grads = tape.gradient(total_loss, vars_to_train)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.opt_g.apply_gradients(zip(grads, vars_to_train))
        
        return {
            'recon_loss': recon_loss,
            'stat_loss': stat_loss,
            'sup_loss': sup_loss
        }
    
    @tf.function
    def train_step_phase2(self, x_real):
        """Phase 2: Adversarial training with focus on temporal dynamics"""
        batch_size = tf.shape(x_real)[0]
        z = tf.random.normal([batch_size, self.config['seq_len'], 
                             self.config['z_dim']])
        
        # Train discriminator
        d_losses = []
        for _ in range(self.config['n_critic']):
            with tf.GradientTape() as tape:
                # Generate synthetic
                e_fake = self.generator(z, training=True)
                h_fake = self.supervisor(e_fake, training=True)
                x_fake = self.recovery(h_fake, training=True)
                
                # Embed real
                h_real = self.embedder(x_real, training=True)
                
                # Discriminator outputs
                d_real = self.discriminator(h_real, training=True)
                d_fake = self.discriminator(h_fake, training=True)
                
                # WGAN-GP loss
                d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
                
                # Gradient penalty
                epsilon = tf.random.uniform([batch_size, 1, 1])
                interpolated = epsilon * h_real + (1 - epsilon) * h_fake
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    d_interpolated = self.discriminator(interpolated, training=True)
                gradients = gp_tape.gradient(d_interpolated, interpolated)
                grad_norm = tf.sqrt(tf.reduce_sum(gradients**2, axis=[1, 2]) + 1e-8)
                gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
                
                total_d_loss = d_loss + self.config['lambda_gp'] * gp
            
            d_grads = tape.gradient(total_d_loss, 
                                   self.discriminator.trainable_variables)
            d_grads = [tf.clip_by_norm(g, 0.5) for g in d_grads]
            self.opt_d.apply_gradients(zip(d_grads, 
                                          self.discriminator.trainable_variables))
            d_losses.append(d_loss)
        
        # Train generator
        with tf.GradientTape() as tape:
            e_fake = self.generator(z, training=True)
            h_fake = self.supervisor(e_fake, training=True)
            x_fake = self.recovery(h_fake, training=True)
            
            d_fake = self.discriminator(h_fake, training=False)
            g_adv_loss = -tf.reduce_mean(d_fake)
            
            # Embed real for supervised loss
            h_real = self.embedder(x_real, training=True)
            h_hat = self.supervisor(h_real, training=True)
            g_sup_loss = tf.reduce_mean(tf.abs(h_real[:, 1:] - h_hat[:, :-1]))
            
            # Reconstruction loss
            x_recon = self.recovery(h_real, training=True)
            g_recon_loss = tf.reduce_mean(tf.abs(x_real - x_recon))
            
            # Statistical matching
            g_stat_loss = self.compute_statistical_loss(x_real, x_fake)
            
            total_g_loss = (g_adv_loss + 
                          self.config['lambda_sup'] * g_sup_loss +
                          self.config['lambda_rec'] * g_recon_loss +
                          self.config['lambda_stat'] * g_stat_loss)
        
        g_vars = (self.generator.trainable_variables + 
                 self.supervisor.trainable_variables +
                 self.recovery.trainable_variables)
        g_grads = tape.gradient(total_g_loss, g_vars)
        g_grads = [tf.clip_by_norm(g, 0.5) for g in g_grads]
        self.opt_g.apply_gradients(zip(g_grads, g_vars))
        
        return {
            'd_loss': tf.reduce_mean(d_losses),
            'g_loss': total_g_loss,
            'adv_loss': g_adv_loss,
            'stat_loss': g_stat_loss
        }
    
    def compute_fid_score(self, real_samples, fake_samples, n_samples=1000):
        """Compute Frechet Inception Distance for time series"""
        # Use PCA features as a proxy for FID
        from sklearn.decomposition import PCA
        from scipy.linalg import sqrtm
        
        # Sample subsets
        idx_real = np.random.choice(len(real_samples), 
                                   min(n_samples, len(real_samples)), 
                                   replace=False)
        idx_fake = np.random.choice(len(fake_samples), 
                                   min(n_samples, len(fake_samples)), 
                                   replace=False)
        
        real_subset = real_samples[idx_real].reshape(-1, real_samples.shape[2])
        fake_subset = fake_samples[idx_fake].reshape(-1, fake_samples.shape[2])
        
        # Compute features via PCA
        pca = PCA(n_components=50)
        combined = np.vstack([real_subset, fake_subset])
        pca.fit(combined)
        
        real_features = pca.transform(real_subset)
        fake_features = pca.transform(fake_subset)
        
        # Compute FID
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features.T)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features.T)
        
        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2*covmean)
        return fid
    
    def validate(self, val_data):
        """Comprehensive validation"""
        # Generate synthetic samples
        n_val = min(1000, len(val_data))
        val_batch = val_data[:n_val]
        
        z = tf.random.normal([n_val, self.config['seq_len'], 
                             self.config['z_dim']])
        
        e_fake = self.generator(z, training=False)
        h_fake = self.supervisor(e_fake, training=False)
        x_fake = self.recovery(h_fake, training=False).numpy()
        
        # Compute multiple metrics
        metrics = {}
        
        # 1. FID score
        metrics['fid'] = self.compute_fid_score(val_batch, x_fake)
        
        # 2. Statistical similarity
        real_flat = val_batch.reshape(-1, val_batch.shape[2])
        fake_flat = x_fake.reshape(-1, x_fake.shape[2])
        
        for i in range(min(10, val_batch.shape[2])):
            real_feat = real_flat[:, i]
            fake_feat = fake_flat[:, i]
            
            # Remove outliers
            real_feat = real_feat[(real_feat > np.percentile(real_feat, 1)) & 
                                 (real_feat < np.percentile(real_feat, 99))]
            fake_feat = fake_feat[(fake_feat > np.percentile(fake_feat, 1)) & 
                                 (fake_feat < np.percentile(fake_feat, 99))]
            
            # Mean and std similarity
            mean_diff = np.abs(np.mean(real_feat) - np.mean(fake_feat))
            std_diff = np.abs(np.std(real_feat) - np.std(fake_feat))
            
            metrics[f'feat_{i}_mean_diff'] = float(mean_diff)
            metrics[f'feat_{i}_std_diff'] = float(std_diff)
        
        # 3. Temporal metrics
        # Autocorrelation similarity
        def compute_autocorr(sequences, max_lag=10):
            acs = []
            for seq in sequences[:, :, 0]:  # Use first feature (price)
                for lag in range(1, max_lag+1):
                    if len(seq) > lag:
                        corr = np.corrcoef(seq[:-lag], seq[lag:])[0, 1]
                        acs.append(np.abs(corr))
            return np.mean(acs) if acs else 0
        
        real_ac = compute_autocorr(val_batch)
        fake_ac = compute_autocorr(x_fake)
        metrics['autocorr_diff'] = float(np.abs(real_ac - fake_ac))
        
        return metrics
    
    def train(self, train_data, val_data, epochs=500):
        """Main training loop with curriculum learning"""
        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        dataset = dataset.shuffle(10000).batch(self.config['batch_size']).prefetch(2)
        
        print("Starting curriculum training...")
        
        # Phase 1: Reconstruction and basic patterns (50 epochs)
        print("\n=== Phase 1: Pattern Learning ===")
        for epoch in range(50):
            losses = []
            for batch in dataset:
                loss_dict = self.train_step_phase1(batch)
                losses.append(loss_dict)
            
            if epoch % 10 == 0:
                avg_loss = {k: np.mean([l[k] for l in losses]) for k in losses[0].keys()}
                print(f"Phase 1 Epoch {epoch}: {avg_loss}")
        
        # Phase 2: Adversarial refinement
        print("\n=== Phase 2: Adversarial Refinement ===")
        for epoch in range(epochs):
            d_losses, g_losses = [], []
            
            for batch in dataset:
                loss_dict = self.train_step_phase2(batch)
                d_losses.append(loss_dict['d_loss'])
                g_losses.append(loss_dict['g_loss'])
            
            # Validation and checkpointing
            if epoch % 5 == 0:
                metrics = self.validate(val_data)
                fid = metrics['fid']
                
                print(f"Epoch {epoch}: D_loss={np.mean(d_losses):.4f}, "
                      f"G_loss={np.mean(g_losses):.4f}, FID={fid:.4f}")
                
                # Save best model based on FID
                if fid < self.best_fid:
                    self.best_fid = fid
                    self.patience = 0
                    self.save_checkpoint(epoch, fid)
                    print(f"✓ New best model (FID: {fid:.4f})")
                else:
                    self.patience += 1
                
                # Early stopping
                if self.patience >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"Training complete. Best FID: {self.best_fid:.4f}")
    
    def save_checkpoint(self, epoch, fid):
        """Save model checkpoint"""
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save weights
        self.generator.save_weights(
            os.path.join(checkpoint_dir, f'generator_epoch_{epoch}_fid_{fid:.4f}.h5')
        )
        self.discriminator.save_weights(
            os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}_fid_{fid:.4f}.h5')
        )
        
        # Save training state
        state = {
            'epoch': epoch,
            'best_fid': float(self.best_fid),
            'patience': self.patience
        }
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(state, f)

# Main training script
if __name__ == "__main__":
    # Configuration
    config = {
        'seq_len': 168,
        'feature_dim': 14,  # Adjust based on your features
        'hidden_dim': 128,
        'z_dim': 64,
        'batch_size': 64,
        'n_critic': 5,
        'lambda_gp': 10.0,
        'lambda_sup': 0.2,
        'lambda_rec': 2.0,
        'lambda_stat': 5.0,
        'patience': 20,
        'checkpoint_dir': 'outputs/checkpoints/timegan_enhanced'
    }
    
    # Load data
    train_data = np.load('data/processed/crypto/train.npy')
    val_data = np.load('data/processed/crypto/val.npy')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Initialize and train
    trainer = CurriculumTimeGANTrainer(config)
    trainer.train(train_data, val_data, epochs=300)