# src/train_timegan_adversarial_tf_fixed.py
# FIXED VERSION - Correct feature dimensions and improved training
import os
import time
import pickle
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from timegan_tf import Embedder, Recovery, Generator, Supervisor, Discriminator

# ----------------- Config (FIXED) -----------------
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_wgan_gp_fixed"
os.makedirs(CKPT_DIR, exist_ok=True)

SEQ_LEN = 168
# 🚨 FIXED: Get correct feature dimension from data
FEATURE_DIM = np.load(os.path.join(DATA_DIR, "train.npy")).shape[2]
HIDDEN_DIM = 128
Z_DIM = 32
BATCH_SIZE = 32

# Learning rates
LR_D = 5e-5
LR_G = 1e-4  
LR_E = 2e-4

# Training lengths
EPOCHS_SUPERVISOR = 80
EPOCHS_ADVERSARIAL = 400

# Adversarial balance
N_CRITIC = 2

# Loss weights
LAMBDA_SUP = 0.5
LAMBDA_REC = 1.0

# WGAN-GP hyper
LAMBDA_GP = 30.0

# Validation
VALIDATION_EVAL_EVERY = 5
VAL_MMD_PATIENCE = 12

# ----------------- TensorBoard Setup -----------------
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/wgan_gp/{current_time}'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# ----------------- Learning Rate Schedules -----------------
lr_schedule_d = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_D, decay_steps=500, decay_rate=0.98, staircase=True)
lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_G, decay_steps=500, decay_rate=0.98, staircase=True)
lr_schedule_e = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_E, decay_steps=500, decay_rate=0.98, staircase=True)

# ----------------- Load data -----------------
train = np.load(os.path.join(DATA_DIR, "train.npy"))
val = np.load(os.path.join(DATA_DIR, "val.npy"))
test = np.load(os.path.join(DATA_DIR, "test.npy"))
print("Loaded shapes:", train.shape, val.shape, test.shape)
print(f"Feature dimension: {FEATURE_DIM}")  # Debug info

train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(2000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_ds_full = tf.data.Dataset.from_tensor_slices(val).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# ----------------- Build models -----------------
embedder = Embedder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
recovery = Recovery(hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, num_layers=2)
generator = Generator(z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(hidden_dim=HIDDEN_DIM, num_layers=1)
discriminator = Discriminator(hidden_dim=HIDDEN_DIM, num_layers=1)

# Build (dummy)
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = discriminator(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ----------------- Optimizers -----------------
opt_e = optimizers.Adam(lr_schedule_e, beta_1=0.5, beta_2=0.999)
opt_r = optimizers.Adam(lr_schedule_e, beta_1=0.5, beta_2=0.999)
opt_g = optimizers.Adam(lr_schedule_g, beta_1=0.5, beta_2=0.999)
opt_s = optimizers.Adam(lr_schedule_g, beta_1=0.5, beta_2=0.999)
opt_d = optimizers.Adam(lr_schedule_d, beta_1=0.5, beta_2=0.999)

# ----------------- Training State Management -----------------
class TrainingState:
    """Track training state for better resumption"""
    def __init__(self):
        self.epoch = 0
        self.best_mmd = np.inf
        self.wait = 0
        self.supervisor_epoch = 0
        
    def save(self, path):
        state = {
            'epoch': self.epoch,
            'best_mmd': self.best_mmd,
            'wait': self.wait,
            'supervisor_epoch': self.supervisor_epoch
        }
        with open(path, 'w') as f:
            json.dump(state, f)
            
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.epoch = state['epoch']
            self.best_mmd = state['best_mmd']
            self.wait = state['wait']
            self.supervisor_epoch = state.get('supervisor_epoch', 0)
            return True
        return False

training_state = TrainingState()
state_path = os.path.join(CKPT_DIR, "training_state.json")

# ----------------- Checkpoint Manager -----------------
ckpt = tf.train.Checkpoint(
    embedder=embedder, recovery=recovery,
    generator=generator, supervisor=supervisor,
    discriminator=discriminator,
    opt_e=opt_e, opt_r=opt_r, opt_g=opt_g, opt_s=opt_s, opt_d=opt_d
)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=6)

# ----------------- Robust optimizer-slot initialization helper -----------------
def _init_optimizer_slots(optimizer, var_list):
    """Create optimizer slot variables"""
    if not var_list:
        return
    zero_grads = [tf.zeros_like(v) for v in var_list]
    optimizer.apply_gradients(zip(zero_grads, var_list))

# ----------------- Attempt to restore training -----------------
_init_optimizer_slots(opt_e, embedder.trainable_variables + recovery.trainable_variables)
_init_optimizer_slots(opt_r, embedder.trainable_variables + recovery.trainable_variables)
_init_optimizer_slots(opt_g, generator.trainable_variables + supervisor.trainable_variables + embedder.trainable_variables + recovery.trainable_variables)
_init_optimizer_slots(opt_s, supervisor.trainable_variables)
_init_optimizer_slots(opt_d, discriminator.trainable_variables)

if manager.latest_checkpoint and training_state.load(state_path):
    status = ckpt.restore(manager.latest_checkpoint)
    try:
        status.assert_existing_objects_matched()
        print("Checkpoint fully restored (models + optimizers + slots).")
    except AssertionError:
        status.expect_partial()
        print("Checkpoint restored with partial match (some optimizer slot variables not found).")
    print(f"Resumed training from: {manager.latest_checkpoint}")
    print(f"Resumed at epoch: {training_state.epoch}, best MMD: {training_state.best_mmd:.6f}")
else:
    print("No checkpoint found — starting from scratch.")
    training_state = TrainingState()

# Restore reconstruction pretrain if available
recon_ckpt_dir = "outputs/checkpoints/recon"
if os.path.isdir(recon_ckpt_dir):
    try:
        recon_ckpt = tf.train.Checkpoint(embedder=embedder, recovery=recovery)
        recon_manager = tf.train.CheckpointManager(recon_ckpt, recon_ckpt_dir, max_to_keep=3)
        if recon_manager.latest_checkpoint:
            recon_ckpt.restore(recon_manager.latest_checkpoint).expect_partial()
            print("Restored recon checkpoint:", recon_manager.latest_checkpoint)
    except Exception as e:
        print("No recon checkpoint restored:", e)

# ----------------- Helper functions -----------------
def sample_z(batch_size):
    return tf.random.normal([batch_size, SEQ_LEN, Z_DIM])

def flatten_seq(x):
    return tf.reshape(x, [tf.shape(x)[0], -1])

def median_sigma_heuristic(X):
    Xf = flatten_seq(X)
    diffs = tf.norm(tf.expand_dims(Xf, 1) - tf.expand_dims(Xf, 0), axis=-1)
    vals = tf.boolean_mask(diffs, diffs > 0.0)
    if tf.size(vals) == 0:
        return tf.constant(1.0, dtype=tf.float32)

    vals = tf.cast(tf.reshape(vals, [-1]), tf.float32)
    n = tf.size(vals)
    sorted_vals = tf.sort(vals)

    def _median_odd():
        mid = n // 2
        return sorted_vals[mid]

    def _median_even():
        mid = n // 2
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])

    median = tf.cond(tf.equal(n % 2, 1), _median_odd, _median_even)
    median = tf.maximum(median, 1.0)
    return median

def rbf_kernel(x, y, sigma):
    xx = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    yy = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    xy = tf.matmul(x, y, transpose_b=True)
    d2 = xx - 2.0*xy + tf.transpose(yy)
    return tf.exp(-d2 / (2.0 * sigma * sigma))

def mmd_rbf(X, Y, sigma=None):
    Xf = flatten_seq(X)
    Yf = flatten_seq(Y)
    if sigma is None:
        sigma = median_sigma_heuristic(Xf)
    Kxx = rbf_kernel(Xf, Xf, sigma)
    Kyy = rbf_kernel(Yf, Yf, sigma)
    Kxy = rbf_kernel(Xf, Yf, sigma)
    return tf.reduce_mean(Kxx) + tf.reduce_mean(Kyy) - 2.0 * tf.reduce_mean(Kxy)

def gradient_penalty(real_H, fake_H):
    """Enhanced gradient penalty implementation"""
    batch_size = tf.shape(real_H)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interp = alpha * real_H + (1.0 - alpha) * fake_H
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interp)
        d_interp = discriminator(interp, training=True)
    
    grads = gp_tape.gradient(d_interp, interp)
    grads = tf.reshape(grads, [batch_size, -1])
    
    grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
    gp = tf.reduce_mean(tf.square(grad_norms - 1.0))
    
    return gp

def comprehensive_validation(n_sample=256):
    """More comprehensive validation metrics"""
    val_arr = val[:n_sample] if val.shape[0] >= n_sample else val
    n = val_arr.shape[0]
    Z = tf.random.normal([n, SEQ_LEN, Z_DIM])
    
    with tf.device('/CPU:0'):
        Ehat = generator(Z, training=False)
        Hhat = supervisor(Ehat, training=False)
        Xhat = recovery(Hhat, training=False)
    
    metrics = {}
    
    sigmas = [0.1, 1.0, 10.0]
    for sigma in sigmas:
        mmd_val = mmd_rbf(
            tf.convert_to_tensor(val_arr, dtype=tf.float32), 
            tf.convert_to_tensor(Xhat, dtype=tf.float32),
            sigma=sigma
        )
        metrics[f'mmd_sigma_{sigma}'] = float(mmd_val.numpy())
    
    real_flat = val_arr.reshape(-1, FEATURE_DIM)
    synth_flat = Xhat.numpy().reshape(-1, FEATURE_DIM)
    
    mean_diffs = []
    std_diffs = []
    
    for i in range(FEATURE_DIM):
        real_mean = np.mean(real_flat[:, i])
        real_std = np.std(real_flat[:, i])
        synth_mean = np.mean(synth_flat[:, i])
        synth_std = np.std(synth_flat[:, i])
        
        mean_diffs.append(abs(real_mean - synth_mean))
        std_diffs.append(abs(real_std - synth_std))
    
    metrics['mean_diff_per_feature'] = float(np.mean(mean_diffs))
    metrics['std_diff_per_feature'] = float(np.mean(std_diffs))
    
    real_range = np.percentile(real_flat, [1, 99])
    synth_range = np.percentile(synth_flat, [1, 99])
    metrics['range_similarity'] = float(1.0 - abs(real_range[1] - synth_range[1]) / (abs(real_range[1]) + 1e-8))
    
    return metrics, Xhat.numpy()

# ----------------- Training steps (WGAN-GP) -----------------
@tf.autograph.experimental.do_not_convert
@tf.function
def supervised_step(x):
    with tf.GradientTape() as tape:
        H = embedder(x, training=False)
        H_hat = supervisor(H, training=True)
        loss_s = tf.reduce_mean(tf.keras.losses.MSE(H[:,1:,:], H_hat[:, :-1, :]))
    grads = tape.gradient(loss_s, supervisor.trainable_variables)
    opt_s.apply_gradients(zip(grads, supervisor.trainable_variables))
    return loss_s

@tf.autograph.experimental.do_not_convert
@tf.function
def critic_step(x):
    batch_size = tf.shape(x)[0]
    z = sample_z(batch_size)
    with tf.GradientTape() as tape:
        H_real = embedder(x, training=False)
        E_hat = generator(z, training=False)
        H_fake = supervisor(E_hat, training=False)

        D_real = discriminator(H_real, training=True)
        D_fake = discriminator(H_fake, training=True)

        loss_critic = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
        gp = gradient_penalty(H_real, H_fake)
        loss = loss_critic + LAMBDA_GP * gp

    grads = tape.gradient(loss, discriminator.trainable_variables)
    opt_d.apply_gradients(zip(grads, discriminator.trainable_variables))
    
    return loss_critic, gp

@tf.autograph.experimental.do_not_convert
@tf.function
def generator_step(x):
    batch_size = tf.shape(x)[0]
    z = sample_z(batch_size)
    with tf.GradientTape(persistent=True) as tape:
        E_hat = generator(z, training=True)
        H_hat = supervisor(E_hat, training=True)
        X_hat = recovery(H_hat, training=True)

        output_scale_penalty = tf.reduce_mean(tf.square(X_hat)) * 100.0
        
        real_mean = tf.reduce_mean(x, axis=[0, 1])
        real_std = tf.math.reduce_std(x, axis=[0, 1])
        synth_mean = tf.reduce_mean(X_hat, axis=[0, 1])
        synth_std = tf.math.reduce_std(X_hat, axis=[0, 1])
        
        mean_matching_loss = tf.reduce_mean(tf.square(real_mean - synth_mean)) * 50.0
        std_matching_loss = tf.reduce_mean(tf.square(real_std - synth_std)) * 50.0
        
        range_penalty = tf.reduce_mean(tf.maximum(tf.abs(X_hat) - 3.0, 0.0)) * 20.0
        
        D_fake = discriminator(H_hat, training=False)
        g_loss_w = -tf.reduce_mean(D_fake)

        H_real = embedder(x, training=False)
        g_loss_s = tf.reduce_mean(tf.keras.losses.MSE(H_real[:,1:,:], H_hat[:, :-1, :]))

        H_enc = embedder(x, training=True)
        X_tilde = recovery(H_hat, training=True)
        g_loss_recon = tf.reduce_mean(tf.keras.losses.MSE(x, X_tilde))

        total_g_loss = (g_loss_w + 
                       LAMBDA_SUP * g_loss_s + 
                       LAMBDA_REC * g_loss_recon + 
                       output_scale_penalty + 
                       mean_matching_loss +
                       std_matching_loss +
                       range_penalty)

    gv = (generator.trainable_variables + supervisor.trainable_variables + 
          embedder.trainable_variables + recovery.trainable_variables)
    grads = tape.gradient(total_g_loss, gv)
    
    grads = [tf.clip_by_norm(g, 0.1) for g in grads]
    
    opt_g.apply_gradients(zip(grads, gv))
    
    del tape
    return total_g_loss, g_loss_w, g_loss_s, g_loss_recon, output_scale_penalty, mean_matching_loss

# ----------------- Training loop -----------------
print("=== Supervisor pretraining ===")
for epoch in range(training_state.supervisor_epoch + 1, EPOCHS_SUPERVISOR + 1):
    t0 = time.time()
    losses = []
    for batch in train_ds:
        l = supervised_step(batch)
        losses.append(l.numpy())
    
    training_state.supervisor_epoch = epoch
    training_state.save(state_path)
    
    with train_summary_writer.as_default():
        tf.summary.scalar('supervisor_loss', np.mean(losses), step=epoch)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"[Sup] Epoch {epoch}/{EPOCHS_SUPERVISOR}  loss={np.mean(losses):.6f}  time={time.time()-t0:.1f}s")

manager.save(checkpoint_number=0)
print("Saved supervisor checkpoint.")

print("=== Adversarial training (WGAN-GP) ===")
best_val_mmd = training_state.best_mmd
wait = training_state.wait

print(f"Adaptive settings: LAMBDA_GP={LAMBDA_GP}, N_CRITIC={N_CRITIC}")

for epoch in range(training_state.epoch + 1, EPOCHS_ADVERSARIAL + 1):
    t0 = time.time()
    d_losses, gp_vals, g_losses, g_w_losses, g_s_losses, g_r_losses, g_scale_penalties, g_mean_penalties = [], [], [], [], [], [], [], []
    
    for batch in train_ds:
        for _ in range(N_CRITIC):
            loss_crit, gp = critic_step(batch)
        
        g_l, g_w, g_s, g_r, g_scale, g_mean = generator_step(batch)

        d_losses.append(float(loss_crit.numpy()))
        gp_vals.append(float(gp.numpy()))
        g_losses.append(float(g_l.numpy()))
        g_w_losses.append(float(g_w.numpy()))
        g_s_losses.append(float(g_s.numpy()))
        g_r_losses.append(float(g_r.numpy()))
        g_scale_penalties.append(float(g_scale.numpy()))
        g_mean_penalties.append(float(g_mean.numpy()))

    with train_summary_writer.as_default():
        tf.summary.scalar('critic_loss', np.mean(d_losses), step=epoch)
        tf.summary.scalar('gradient_penalty', np.mean(gp_vals), step=epoch)
        tf.summary.scalar('generator_total_loss', np.mean(g_losses), step=epoch)
        tf.summary.scalar('generator_w_loss', np.mean(g_w_losses), step=epoch)
        tf.summary.scalar('generator_s_loss', np.mean(g_s_losses), step=epoch)
        tf.summary.scalar('generator_r_loss', np.mean(g_r_losses), step=epoch)
        tf.summary.scalar('generator_scale_penalty', np.mean(g_scale_penalties), step=epoch)
        tf.summary.scalar('generator_mean_penalty', np.mean(g_mean_penalties), step=epoch)
        
        tf.summary.scalar('learning_rate_d', lr_schedule_d(opt_d.iterations), step=epoch)
        tf.summary.scalar('learning_rate_g', lr_schedule_g(opt_g.iterations), step=epoch)
        tf.summary.scalar('learning_rate_e', lr_schedule_e(opt_e.iterations), step=epoch)

    print(f"[WGAN-GP] Epoch {epoch}/{EPOCHS_ADVERSARIAL}  "
          f"Critic={np.mean(d_losses):.6f}  GP={np.mean(gp_vals):.6f}  "
          f"G={np.mean(g_losses):.6f}  ScalePen={np.mean(g_scale_penalties):.6f}")

# Save final weights
embedder.save_weights(os.path.join(CKPT_DIR, "embedder_final.weights.h5"))
recovery.save_weights(os.path.join(CKPT_DIR, "recovery_final.weights.h5"))
generator.save_weights(os.path.join(CKPT_DIR, "generator_final.weights.h5"))
supervisor.save_weights(os.path.join(CKPT_DIR, "supervisor_final.weights.h5"))
discriminator.save_weights(os.path.join(CKPT_DIR, "discriminator_final.weights.h5"))
print("Saved final weights.")

# ----------------- Enhanced Generation Helper -----------------
def generate(n_samples, inverse_scale=True, return_components=False):
    """Generate samples with quality metrics"""
    Z = tf.random.normal([n_samples, SEQ_LEN, Z_DIM])
    Ehat = generator(Z, training=False)
    Hhat = supervisor(Ehat, training=False)
    Xhat = recovery(Hhat, training=False).numpy()

    if not np.isfinite(Xhat).all():
        print("Warning: Generated data contains non-finite values!")
        Xhat = np.nan_to_num(Xhat)

    if not inverse_scale:
        if return_components:
            return Xhat, Ehat.numpy(), Hhat.numpy()
        return Xhat

    scalers_path = os.path.join(DATA_DIR, "scalers.pkl")
    if not os.path.exists(scalers_path):
        scalers_path = os.path.join(DATA_DIR, "scaler.pkl")
    
    if os.path.exists(scalers_path):
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        n, T, D = Xhat.shape
        X_inv = np.empty_like(Xhat)
        if isinstance(scalers, list):
            for d, s in enumerate(scalers):
                X_inv[:, :, d] = s.inverse_transform(Xhat[:, :, d].reshape(-1, 1)).reshape(n, T)
        else:
            X_inv = scalers.inverse_transform(Xhat.reshape(-1, D)).reshape(Xhat.shape)
        
        if return_components:
            return X_inv, Ehat.numpy(), Hhat.numpy()
        return X_inv
    else:
        print("No scalers found; returning scaled synthetic samples.")
        if return_components:
            return Xhat, Ehat.numpy(), Hhat.numpy()
        return Xhat

def generate_with_metrics(n_samples, inverse_scale=True):
    """Generate samples with comprehensive quality metrics"""
    synthetic_data = generate(n_samples, inverse_scale)
    
    metrics = {}
    metrics['finite_ratio'] = np.isfinite(synthetic_data).mean()
    metrics['value_range'] = (synthetic_data.min(), synthetic_data.max())
    metrics['mean'] = synthetic_data.mean()
    metrics['std'] = synthetic_data.std()
    
    unique_samples = len(np.unique(synthetic_data.reshape(n_samples, -1), axis=0))
    metrics['diversity_ratio'] = unique_samples / n_samples
    
    print("Generation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v[0]:.4f} to {v[1]:.4f}")
        else:
            print(f"  {k}: {v:.4f}")
    
    return synthetic_data, metrics

if __name__ == "__main__":
    s, metrics = generate_with_metrics(5)
    print("Generated sample shape:", s.shape)
    
    s_full, e_hat, h_hat = generate(5, return_components=True)
    print("Component shapes - Synthetic:", s_full.shape, 
          "E_hat:", e_hat.shape, "H_hat:", h_hat.shape)
    
    print("Training completed successfully!")
    print(f"TensorBoard logs: tensorboard --logdir={train_log_dir}")