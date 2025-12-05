import os
import time
import pickle
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import sys
sys.path.append('..')
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

# ----------------- Enhanced Configuration -----------------
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_enhanced"
os.makedirs(CKPT_DIR, exist_ok=True)

# Auto-detect dimensions
train_files = [f for f in os.listdir(DATA_DIR) if f.startswith('train') and f.endswith('.npy')]
if train_files:
    train_path = os.path.join(DATA_DIR, train_files[0])
    train_sample = np.load(train_path)
    SEQ_LEN = train_sample.shape[1]
    FEATURE_DIM = train_sample.shape[2]
    print(f"🎯 Detected: SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")
else:
    SEQ_LEN = 168
    FEATURE_DIM = 14
    print(f"⚠️ Using defaults: SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")

# ENHANCED parameters matching your reconstruction training
HIDDEN_DIM = 256    # Match enhanced reconstruction
Z_DIM = 64          # Increased for better variation
BATCH_SIZE = 64     # Balanced for stability
NUM_LAYERS = 3      # Match enhanced reconstruction

# Enhanced Learning rates
LR_D = 1e-4
LR_G = 2e-4  
LR_E = 2e-4

# Extended training for better convergence
EPOCHS_SUPERVISOR = 50     # Extended
EPOCHS_ADVERSARIAL = 500   # Extended for convergence

# Training balance
N_CRITIC = 2

# Enhanced Loss weights
LAMBDA_SUP = 0.2
LAMBDA_REC = 1.0
LAMBDA_GP = 10.0
LAMBDA_MMD = 1.0

# Validation
VALIDATION_EVAL_EVERY = 10
VAL_PATIENCE = 30

# ----------------- Enhanced TensorBoard Setup -----------------
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/timegan_enhanced/{current_time}'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# ----------------- Enhanced Data Loading -----------------
def load_enhanced_data(data_dir):
    """Load data with better validation"""
    train_data = []
    val_data = []
    
    for split in ['train', 'val']:
        split_files = [f for f in os.listdir(data_dir) if f.startswith(split) and f.endswith('.npy')]
        for file in split_files:
            data = np.load(os.path.join(data_dir, file))
            if split == 'train':
                train_data.append(data)
            elif split == 'val':
                val_data.append(data)
    
    if train_data:
        train_all = np.concatenate(train_data, axis=0)
        print(f"📊 Enhanced Training Data: {train_all.shape}")
        
        # Data quality check
        print("🔍 Data Statistics:")
        for i in range(min(5, train_all.shape[2])):
            feature_data = train_all[:, :, i].flatten()
            print(f"  Feature {i}: mean={feature_data.mean():.4f}, std={feature_data.std():.4f}")
            
    else:
        raise ValueError("No training data found!")
    
    if val_data:
        val_all = np.concatenate(val_data, axis=0)
    else:
        val_all = train_all[:100]  # Reasonable validation set
    
    return train_all, val_all

print("🔄 Loading enhanced data...")
train, val = load_enhanced_data(DATA_DIR)

# Enhanced dataset pipeline
train_ds = tf.data.Dataset.from_tensor_slices(train)
train_ds = train_ds.shuffle(min(5000, train.shape[0]))  # Better shuffling
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# ----------------- Build Enhanced Models with Pre-trained Weights -----------------
print("🔨 Building enhanced models with pre-trained reconstruction...")

# Build models with enhanced architecture
embedder = Embedder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=NUM_LAYERS, dropout=0.2)
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=2, dropout=0.2)
discriminator = Discriminator(hidden_dim=HIDDEN_DIM, num_layers=2, dropout=0.2)

# Build models
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = discriminator(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

print(f"🎯 Enhanced Model Parameters:")
print(f"  Embedder: {sum(np.prod(v.shape) for v in embedder.trainable_variables):,}")
print(f"  Recovery: {sum(np.prod(v.shape) for v in recovery.trainable_variables):,}")
print(f"  Generator: {sum(np.prod(v.shape) for v in generator.trainable_variables):,}")
print(f"  Supervisor: {sum(np.prod(v.shape) for v in supervisor.trainable_variables):,}")
print(f"  Discriminator: {sum(np.prod(v.shape) for v in discriminator.trainable_variables):,}")

# ----------------- Load Pre-trained Reconstruction Weights -----------------
def load_pretrained_weights():
    """Load pre-trained reconstruction model weights"""
    recon_checkpoint_dir = "outputs/checkpoints/recon_enhanced_fixed"
    
    embedder_path = os.path.join(recon_checkpoint_dir, "enhanced_embedder_final.h5")
    recovery_path = os.path.join(recon_checkpoint_dir, "enhanced_recovery_final.h5")
    
    if os.path.exists(embedder_path) and os.path.exists(recovery_path):
        embedder.load_weights(embedder_path)
        recovery.load_weights(recovery_path)
        print("✅ Loaded pre-trained reconstruction weights")
        return True
    else:
        print("⚠️ No pre-trained reconstruction weights found")
        print("💡 Please run train_recon_enhanced_fixed.py first")
        return False

# Load pre-trained weights
if not load_pretrained_weights():
    print("🚀 Continuing without pre-trained weights (cold start)")

# ----------------- Enhanced Optimizers -----------------
opt_e = optimizers.Adam(LR_E, beta_1=0.5, beta_2=0.9)
opt_r = optimizers.Adam(LR_E, beta_1=0.5, beta_2=0.9)
opt_g = optimizers.Adam(LR_G, beta_1=0.5, beta_2=0.9)
opt_s = optimizers.Adam(LR_G, beta_1=0.5, beta_2=0.9)
opt_d = optimizers.Adam(LR_D, beta_1=0.5, beta_2=0.9)

# ----------------- Enhanced Training State -----------------
class EnhancedTrainingState:
    def __init__(self):
        self.epoch = 0
        self.best_validation_loss = np.inf
        self.wait = 0
        self.supervisor_epoch = 0
        self.training_history = []
        
    def save(self, path):
        state = {
            'epoch': self.epoch,
            'best_validation_loss': self.best_validation_loss,
            'wait': self.wait,
            'supervisor_epoch': self.supervisor_epoch,
            'training_history': self.training_history
        }
        with open(path, 'w') as f:
            json.dump(state, f)
            
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.epoch = state['epoch']
            self.best_validation_loss = state['best_validation_loss']
            self.wait = state['wait']
            self.supervisor_epoch = state.get('supervisor_epoch', 0)
            self.training_history = state.get('training_history', [])
            return True
        return False

training_state = EnhancedTrainingState()
state_path = os.path.join(CKPT_DIR, "enhanced_training_state.json")

# ----------------- Enhanced Checkpoint Manager -----------------
ckpt = tf.train.Checkpoint(
    embedder=embedder, recovery=recovery,
    generator=generator, supervisor=supervisor,
    discriminator=discriminator,
    opt_e=opt_e, opt_r=opt_r, opt_g=opt_g, opt_s=opt_s, opt_d=opt_d
)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

# ----------------- Enhanced Validation -----------------
def enhanced_validation():
    """Comprehensive validation with multiple metrics"""
    val_sample = val[:64]  # Fixed size for consistency
    n = val_sample.shape[0]
    Z = tf.random.normal([n, SEQ_LEN, Z_DIM], stddev=0.5)
    
    # Generate synthetic data
    E_hat = generator(Z, training=False)
    H_hat = supervisor(E_hat, training=False)
    X_hat = recovery(H_hat, training=False)
    
    # Multiple validation metrics
    # 1. Reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.MSE(val_sample, X_hat))
    
    # 2. Statistical differences
    real_means = tf.reduce_mean(val_sample, axis=[0, 1])
    synth_means = tf.reduce_mean(X_hat, axis=[0, 1])
    mean_diff = tf.reduce_mean(tf.abs(real_means - synth_means))
    
    real_stds = tf.math.reduce_std(val_sample, axis=[0, 1])
    synth_stds = tf.math.reduce_std(X_hat, axis=[0, 1])
    std_diff = tf.reduce_mean(tf.abs(real_stds - synth_stds))
    
    # 3. Feature correlation preservation (simplified)
    real_flat = tf.reshape(val_sample, [-1, FEATURE_DIM])
    synth_flat = tf.reshape(X_hat, [-1, FEATURE_DIM])
    
    real_corr = tf.linalg.norm(tf.linalg.matmul(real_flat, real_flat, transpose_a=True))
    synth_corr = tf.linalg.norm(tf.linalg.matmul(synth_flat, synth_flat, transpose_a=True))
    corr_similarity = 1.0 - tf.abs(real_corr - synth_corr) / real_corr
    
    validation_score = 0.4 * (1.0 - reconstruction_loss) + 0.3 * (1.0 - mean_diff) + 0.3 * corr_similarity
    
    return {
        'reconstruction_loss': float(reconstruction_loss.numpy()),
        'mean_diff': float(mean_diff.numpy()),
        'std_diff': float(std_diff.numpy()),
        'corr_similarity': float(corr_similarity.numpy()),
        'validation_score': float(validation_score.numpy()),
        'synthetic_data': X_hat.numpy()
    }

# ----------------- Enhanced Training Steps -----------------
@tf.function
def enhanced_supervised_step(x):
    with tf.GradientTape() as tape:
        H = embedder(x, training=True)
        H_hat = supervisor(H, training=True)
        # Enhanced supervisor loss with temporal consistency
        loss_s = tf.reduce_mean(tf.keras.losses.MSE(H[:,1:,:], H_hat[:, :-1, :]))
        # Add regularization
        loss_s += 0.01 * tf.reduce_sum([tf.nn.l2_loss(v) for v in supervisor.trainable_variables])
    grads = tape.gradient(loss_s, supervisor.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    opt_s.apply_gradients(zip(grads, supervisor.trainable_variables))
    return loss_s

@tf.function
def enhanced_critic_step(x):
    batch_size = tf.shape(x)[0]
    z = tf.random.normal([batch_size, SEQ_LEN, Z_DIM], stddev=0.5)
    
    with tf.GradientTape() as tape:
        H_real = embedder(x, training=False)
        E_hat = generator(z, training=False)
        H_fake = supervisor(E_hat, training=False)

        D_real = discriminator(H_real, training=True)
        D_fake = discriminator(H_fake, training=True)

        loss_critic = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
        
        # Enhanced gradient penalty
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interp = alpha * H_real + (1.0 - alpha) * H_fake
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interp)
            d_interp = discriminator(interp, training=True)
        grads = gp_tape.gradient(d_interp, interp)
        grads = tf.reshape(grads, [batch_size, -1])
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean(tf.square(grad_norms - 1.0))
        
        loss = loss_critic + LAMBDA_GP * gp

    grads = tape.gradient(loss, discriminator.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    opt_d.apply_gradients(zip(grads, discriminator.trainable_variables))
    
    return loss_critic, gp

@tf.function
def enhanced_generator_step(x):
    batch_size = tf.shape(x)[0]
    z = tf.random.normal([batch_size, SEQ_LEN, Z_DIM], stddev=0.5)
    
    with tf.GradientTape(persistent=True) as tape:
        E_hat = generator(z, training=True)
        H_hat = supervisor(E_hat, training=True)
        X_hat = recovery(H_hat, training=True)

        # Enhanced statistical matching
        real_means = tf.reduce_mean(x, axis=[0, 1])
        synth_means = tf.reduce_mean(X_hat, axis=[0, 1])
        mean_matching_loss = tf.reduce_mean(tf.square(real_means - synth_means))
        
        real_stds = tf.math.reduce_std(x, axis=[0, 1])
        synth_stds = tf.math.reduce_std(X_hat, axis=[0, 1])
        std_matching_loss = tf.reduce_mean(tf.square(real_stds - synth_stds))
        
        # Adversarial loss
        D_fake = discriminator(H_hat, training=False)
        g_loss_w = -tf.reduce_mean(D_fake)

        # Supervisor loss
        H_real = embedder(x, training=False)
        g_loss_s = tf.reduce_mean(tf.keras.losses.MSE(H_real[:,1:,:], H_hat[:, :-1, :]))

        # Reconstruction loss
        g_loss_recon = tf.reduce_mean(tf.keras.losses.MSE(x, X_hat))

        # Enhanced Total loss with better balancing
        total_g_loss = (
            g_loss_w + 
            LAMBDA_SUP * g_loss_s + 
            LAMBDA_REC * g_loss_recon + 
            5.0 * mean_matching_loss +
            5.0 * std_matching_loss
        )

    gv = generator.trainable_variables + supervisor.trainable_variables
    grads = tape.gradient(total_g_loss, gv)
    grads = [tf.clip_by_norm(g, 0.1) for g in grads]
    opt_g.apply_gradients(zip(grads, gv))
    
    del tape
    return total_g_loss, g_loss_w, g_loss_s, g_loss_recon, mean_matching_loss, std_matching_loss

# ----------------- Enhanced Training Loop -----------------
def run_enhanced_training():
    print("🚀 ENHANCED TIMEGAN TRAINING STARTED")
    print("=" * 60)
    print(f"📊 Data: {train.shape}")
    print(f"🎯 Target: High-quality synthetic data (90%+ similarity)")
    print(f"⏰ Epochs: Supervisor={EPOCHS_SUPERVISOR}, Adversarial={EPOCHS_ADVERSARIAL}")
    print("=" * 60)
    
    # Clear memory
    tf.keras.backend.clear_session()
    
    # Enhanced Supervisor pretraining
    if training_state.supervisor_epoch < EPOCHS_SUPERVISOR:
        print("🎓 ENHANCED SUPERVISOR PRETRAINING")
        for epoch in range(training_state.supervisor_epoch + 1, EPOCHS_SUPERVISOR + 1):
            t0 = time.time()
            losses = []
            for batch in train_ds:
                l = enhanced_supervised_step(batch)
                losses.append(l.numpy())
            
            training_state.supervisor_epoch = epoch
            training_state.save(state_path)
            
            with train_summary_writer.as_default():
                tf.summary.scalar('enhanced_supervisor_loss', np.mean(losses), step=epoch)
            
            if epoch % 10 == 0 or epoch <= 5:
                print(f"[Sup] Epoch {epoch:3d}/{EPOCHS_SUPERVISOR}  loss={np.mean(losses):.6f}  time={time.time()-t0:.1f}s")

        manager.save(checkpoint_number=0)
        print("✅ Enhanced supervisor pretraining completed")

    # Enhanced Adversarial training
    print("⚔️  ENHANCED ADVERSARIAL TRAINING (WGAN-GP)")
    print(f"🔧 Settings: LAMBDA_GP={LAMBDA_GP}, N_CRITIC={N_CRITIC}")

    best_validation_score = training_state.best_validation_loss
    wait = training_state.wait

    for epoch in range(training_state.epoch + 1, EPOCHS_ADVERSARIAL + 1):
        t0 = time.time()
        d_losses, gp_vals, g_losses = [], [], []
        g_w_losses, g_s_losses, g_r_losses, g_mean_losses, g_std_losses = [], [], [], [], []
        
        # Memory management
        if epoch % 20 == 0:
            tf.keras.backend.clear_session()
        
        for batch in train_ds:
            # Train critic multiple times
            for _ in range(N_CRITIC):
                loss_crit, gp = enhanced_critic_step(batch)
                d_losses.append(float(loss_crit.numpy()))
                gp_vals.append(float(gp.numpy()))
            
            # Train generator once
            g_l, g_w, g_s, g_r, g_mean, g_std = enhanced_generator_step(batch)
            g_losses.append(float(g_l.numpy()))
            g_w_losses.append(float(g_w.numpy()))
            g_s_losses.append(float(g_s.numpy()))
            g_r_losses.append(float(g_r.numpy()))
            g_mean_losses.append(float(g_mean.numpy()))
            g_std_losses.append(float(g_std.numpy()))

        training_state.epoch = epoch
        
        # Enhanced Validation
        if epoch % VALIDATION_EVAL_EVERY == 0:
            val_metrics = enhanced_validation()
            val_score = val_metrics['validation_score']
            
            print(f"📊 Enhanced Validation - Score: {val_score:.4f}, "
                  f"Recon: {val_metrics['reconstruction_loss']:.4f}, "
                  f"Mean Diff: {val_metrics['mean_diff']:.4f}")
            
            with train_summary_writer.as_default():
                tf.summary.scalar('validation_score', val_score, step=epoch)
                tf.summary.scalar('validation_reconstruction', val_metrics['reconstruction_loss'], step=epoch)
                tf.summary.scalar('validation_mean_diff', val_metrics['mean_diff'], step=epoch)
                tf.summary.scalar('validation_corr_similarity', val_metrics['corr_similarity'], step=epoch)
            
            # Enhanced early stopping
            if val_score > best_validation_score:
                best_validation_score = val_score
                wait = 0
                manager.save(checkpoint_number=epoch)
                print(f"🎯 NEW BEST! Validation Score: {best_validation_score:.4f}, checkpoint saved.")
                
                # Quick generation test
                test_gen = generate_quick_sample(10)
                print(f"🧪 Quick generation test: {test_gen.shape}")
                
            else:
                wait += 1
                if wait >= VAL_PATIENCE:
                    print(f"🛑 Early stopping at epoch {epoch}. Best Validation Score: {best_validation_score:.4f}")
                    break
        
        training_state.best_validation_loss = best_validation_score
        training_state.wait = wait
        training_state.save(state_path)
        
        # Enhanced logging
        with train_summary_writer.as_default():
            tf.summary.scalar('enhanced_critic_loss', np.mean(d_losses), step=epoch)
            tf.summary.scalar('enhanced_gradient_penalty', np.mean(gp_vals), step=epoch)
            tf.summary.scalar('enhanced_generator_total_loss', np.mean(g_losses), step=epoch)
            tf.summary.scalar('enhanced_generator_w_loss', np.mean(g_w_losses), step=epoch)
            tf.summary.scalar('enhanced_generator_s_loss', np.mean(g_s_losses), step=epoch)
            tf.summary.scalar('enhanced_generator_r_loss', np.mean(g_r_losses), step=epoch)
            tf.summary.scalar('enhanced_generator_mean_loss', np.mean(g_mean_losses), step=epoch)
            tf.summary.scalar('enhanced_generator_std_loss', np.mean(g_std_losses), step=epoch)

        # Enhanced progress reporting
        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch <= 10:
            print(f"[WGAN-GP] Epoch {epoch:3d}/{EPOCHS_ADVERSARIAL}  "
                  f"C={np.mean(d_losses):7.3f}  GP={np.mean(gp_vals):6.4f}  "
                  f"G={np.mean(g_losses):7.3f}  Val={best_validation_score:.4f}  Time={elapsed:.1f}s")

    return best_validation_score

def generate_quick_sample(n_samples):
    """Quick generation for testing"""
    Z = tf.random.normal([n_samples, SEQ_LEN, Z_DIM], stddev=0.5)
    E_hat = generator(Z, training=False)
    H_hat = supervisor(E_hat, training=False)
    X_hat = recovery(H_hat, training=False)
    return X_hat.numpy()

# ----------------- Generation Function -----------------
def generate_synthetic(n_samples, batch_size=32):
    """Memory-efficient generation"""
    synthetic_batches = []
    
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        Z = tf.random.normal([current_batch, SEQ_LEN, Z_DIM], stddev=0.5)
        E_hat = generator(Z, training=False)
        H_hat = supervisor(E_hat, training=False)
        X_hat = recovery(H_hat, training=False).numpy()
        synthetic_batches.append(X_hat)
    
    return np.concatenate(synthetic_batches, axis=0)

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    print("🧠 ENHANCED TIMEGAN CONFIGURATION:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Hidden Dim: {HIDDEN_DIM}")
    print(f"   Latent Dim: {Z_DIM}")
    print(f"   Model Layers: {NUM_LAYERS}")
    print(f"   Learning Rates: G={LR_G}, D={LR_D}")
    
    # Restore checkpoint if exists
    if manager.latest_checkpoint and training_state.load(state_path):
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"✅ Resumed from: {manager.latest_checkpoint}")
        print(f"📊 Epoch: {training_state.epoch}, Best Val Score: {training_state.best_validation_loss:.4f}")
    else:
        print("🚀 Starting fresh enhanced training")
    
    best_score = run_enhanced_training()
    
    # Save final enhanced models
    print("💾 Saving final enhanced models...")
    embedder.save_weights(os.path.join(CKPT_DIR, "enhanced_embedder_final.weights.h5"))
    recovery.save_weights(os.path.join(CKPT_DIR, "enhanced_recovery_final.weights.h5"))
    generator.save_weights(os.path.join(CKPT_DIR, "enhanced_generator_final.weights.h5"))
    supervisor.save_weights(os.path.join(CKPT_DIR, "enhanced_supervisor_final.weights.h5"))
    discriminator.save_weights(os.path.join(CKPT_DIR, "enhanced_discriminator_final.weights.h5"))
    
    # Final generation test
    print("\n🧪 Final generation test...")
    final_sample = generate_synthetic(100)
    print(f"✅ Final sample shape: {final_sample.shape}")
    
    print("🎉 ENHANCED TIMEGAN TRAINING COMPLETED!")
    print(f"🎯 Final best validation score: {best_score:.4f}")
    print(f"📈 TensorBoard: tensorboard --logdir={train_log_dir}")
    print(f"💾 Models saved to: {CKPT_DIR}")