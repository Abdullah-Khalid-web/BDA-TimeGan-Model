import os, pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from timegan_tf import Embedder, Recovery

# Enhanced Config
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/recon_enhanced_fixed"
os.makedirs(CKPT_DIR, exist_ok=True)

# Auto-detect dimensions
train = np.load(os.path.join(DATA_DIR, "train.npy"))
SEQ_LEN = train.shape[1]
FEATURE_DIM = train.shape[2]

# ENHANCED parameters for better learning
BATCH_SIZE = 32
HIDDEN_DIM = 256  # Increased capacity
NUM_LAYERS = 3
NUM_EPOCHS = 1000  # Extended training
LR = 1e-4  # Lower learning rate for stability
PATIENCE = 50  # More patience

print("🎯 ENHANCED RECONSTRUCTION TRAINING (FIXED)")
print(f"📊 Data: SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")
print(f"⚙️  Enhanced Params: HIDDEN_DIM={HIDDEN_DIM}, EPOCHS={NUM_EPOCHS}, LR={LR}")

# Load data
val = np.load(os.path.join(DATA_DIR, "val.npy"))
test = np.load(os.path.join(DATA_DIR, "test.npy"))

print("📈 Data Shapes:")
print(f"  Train: {train.shape} ({train.shape[0]:,} sequences)")
print(f"  Val:   {val.shape} ({val.shape[0]:,} sequences)") 

# Enhanced data pipeline with better normalization check
print("\n🔍 Data Statistics Check:")
train_flat = train.reshape(-1, FEATURE_DIM)
for i in range(min(5, FEATURE_DIM)):
    feature_data = train_flat[:, i]
    print(f"  Feature {i}: mean={feature_data.mean():.4f}, std={feature_data.std():.4f}, "
          f"min={feature_data.min():.4f}, max={feature_data.max():.4f}")

train_ds = tf.data.Dataset.from_tensor_slices(train)
train_ds = train_ds.shuffle(min(5000, train.shape[0]))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(val)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Enhanced Model with better initialization
print("🔨 Building enhanced models...")
embedder = Embedder(
    input_dim=FEATURE_DIM, 
    hidden_dim=HIDDEN_DIM, 
    num_layers=NUM_LAYERS, 
    dropout=0.2
)
recovery = Recovery(
    hidden_dim=HIDDEN_DIM, 
    output_dim=FEATURE_DIM, 
    num_layers=NUM_LAYERS, 
    dropout=0.2
)

# Build models
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

print(f"📊 Enhanced Model Parameters:")
print(f"  Embedder: {sum(np.prod(v.shape) for v in embedder.trainable_variables):,}")
print(f"  Recovery: {sum(np.prod(v.shape) for v in recovery.trainable_variables):,}")

# Enhanced Optimizer with gradient clipping
optimizer = optimizers.Adam(
    learning_rate=LR,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

# Enhanced loss function - focus on key features
def weighted_reconstruction_loss(y_true, y_pred):
    # Base MSE loss
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    
    # Additional focus on price features (first 4 features)
    price_features_mse = tf.keras.losses.MSE(y_true[:, :, :4], y_pred[:, :, :4])
    
    # Combine with emphasis on price features
    total_loss = 0.7 * mse_loss + 0.3 * price_features_mse
    
    return total_loss

# Enhanced Checkpoint manager
ckpt = tf.train.Checkpoint(
    optimizer=optimizer, 
    embedder=embedder, 
    recovery=recovery
)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

# Enhanced TensorBoard logging
current_time = str(int(tf.timestamp().numpy()))
train_log_dir = os.path.join(CKPT_DIR, f"logs_{current_time}")
train_writer = tf.summary.create_file_writer(train_log_dir)

@tf.function
def enhanced_train_step(x):
    with tf.GradientTape() as tape:
        h = embedder(x, training=True)
        x_tilde = recovery(h, training=True)
        loss = weighted_reconstruction_loss(x, x_tilde)
        
    vars_ = embedder.trainable_variables + recovery.trainable_variables
    grads = tape.gradient(loss, vars_)
    
    # Gradient clipping for stability
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    
    optimizer.apply_gradients(zip(grads, vars_))
    return loss

@tf.function  
def enhanced_val_step(x):
    h = embedder(x, training=False)
    x_tilde = recovery(h, training=False)
    return weighted_reconstruction_loss(x, x_tilde)

def calculate_detailed_metrics(original, reconstructed):
    """Calculate comprehensive reconstruction metrics - FIXED VERSION"""
    metrics = {}
    
    # Convert to numpy for calculation
    original_np = original.numpy() if hasattr(original, 'numpy') else original
    reconstructed_np = reconstructed.numpy() if hasattr(reconstructed, 'numpy') else reconstructed
    
    # Overall metrics - use numpy directly
    mse = np.mean((original_np - reconstructed_np) ** 2)
    mae = np.mean(np.abs(original_np - reconstructed_np))
    
    metrics['overall_mse'] = float(mse)
    metrics['overall_mae'] = float(mae)
    
    # Feature-wise metrics
    metrics['feature_mse'] = []
    metrics['feature_mae'] = []
    
    for i in range(FEATURE_DIM):
        orig_feat = original_np[:, :, i]
        rec_feat = reconstructed_np[:, :, i]
        feature_mse = np.mean((orig_feat - rec_feat) ** 2)
        feature_mae = np.mean(np.abs(orig_feat - rec_feat))
        
        metrics['feature_mse'].append(float(feature_mse))
        metrics['feature_mae'].append(float(feature_mae))
    
    return metrics

print("🚀 Starting enhanced reconstruction training...")
print("=" * 60)

best_val_loss = float('inf')
wait = 0
training_history = []

# Learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 100:
        return LR
    elif epoch < 300:
        return LR * 0.5
    elif epoch < 600:
        return LR * 0.2
    else:
        return LR * 0.1

for epoch in range(1, NUM_EPOCHS + 1):
    # Update learning rate
    new_lr = lr_scheduler(epoch)
    tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
    
    # Training
    train_losses = []
    for batch in train_ds:
        loss = enhanced_train_step(batch)
        train_losses.append(loss.numpy())
    
    train_loss = float(np.mean(train_losses))
    
    # Validation
    val_losses = []
    for batch in val_ds:
        loss = enhanced_val_step(batch)
        val_losses.append(loss.numpy())
    
    val_loss = float(np.mean(val_losses))
    
    # Calculate detailed metrics every 50 epochs
    price_features_mse = val_loss
    other_features_mse = val_loss
    
    if epoch % 50 == 0 or epoch <= 10:
        try:
            sample_batch = next(iter(val_ds))
            h_sample = embedder(sample_batch, training=False)
            x_rec_sample = recovery(h_sample, training=False)
            metrics = calculate_detailed_metrics(sample_batch, x_rec_sample)
            
            # Log key feature performance
            price_features_mse = np.mean(metrics['feature_mse'][:4])
            other_features_mse = np.mean(metrics['feature_mse'][4:])
        except Exception as e:
            print(f"⚠️ Metrics calculation failed: {e}")
            # Use default values if metrics calculation fails
            price_features_mse = val_loss
            other_features_mse = val_loss
    
    # Store history
    training_history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': new_lr,
        'price_features_mse': price_features_mse,
        'other_features_mse': other_features_mse
    })
    
    # Enhanced logging
    with train_writer.as_default():
        tf.summary.scalar("train_loss", train_loss, step=epoch)
        tf.summary.scalar("val_loss", val_loss, step=epoch)
        tf.summary.scalar("learning_rate", new_lr, step=epoch)
        tf.summary.scalar("price_features_mse", price_features_mse, step=epoch)
        tf.summary.scalar("other_features_mse", other_features_mse, step=epoch)
    
    # Enhanced progress reporting
    if epoch % 20 == 0 or epoch <= 10:
        print(f"[Recon] Epoch {epoch:04d}/{NUM_EPOCHS}: "
              f"train={train_loss:.6f}, val={val_loss:.6f}, "
              f"LR={new_lr:.2e}, PriceMSE={price_features_mse:.6f}")

    # Enhanced early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        manager.save()
        
        # Save best metrics
        best_metrics = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'price_features_mse': price_features_mse,
            'other_features_mse': other_features_mse
        }
        
        with open(os.path.join(CKPT_DIR, "best_metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=2)
            
        if epoch % 50 == 0:
            print(f"  💾 Checkpoint saved (val {best_val_loss:.6f})")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
            break

    # Progress every 100 epochs
    if epoch % 100 == 0:
        print(f"📊 Progress: {epoch}/{NUM_EPOCHS} epochs completed")
        print(f"   Best Val Loss: {best_val_loss:.6f}")
        print(f"   Current LR: {new_lr:.2e}")

print("✅ Enhanced reconstruction training completed!")

# Save final enhanced weights
embedder.save_weights(os.path.join(CKPT_DIR, "enhanced_embedder_final.h5"))
recovery.save_weights(os.path.join(CKPT_DIR, "enhanced_recovery_final.h5"))

print("💾 Saved final enhanced weights")

# Generate comprehensive reconstruction report
print("\n🔍 Generating enhanced reconstruction report...")

# Load feature names
feature_names = []
feature_names_path = os.path.join(DATA_DIR, "features.txt")
if os.path.exists(feature_names_path):
    with open(feature_names_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
else:
    feature_names = [f"feature_{i}" for i in range(FEATURE_DIM)]

# Test reconstruction on validation set
val_sample = val[:10]
h_val = embedder(val_sample, training=False)
x_rec_val = recovery(h_val, training=False).numpy()

# Calculate final metrics
final_metrics = calculate_detailed_metrics(val_sample, x_rec_val)

print(f"📊 FINAL RECONSTRUCTION METRICS:")
print(f"  Overall MSE:  {final_metrics['overall_mse']:.6f}")
print(f"  Overall MAE:  {final_metrics['overall_mae']:.6f}")

print(f"\n🔍 Feature-wise MSE:")
for i, name in enumerate(feature_names):
    mse = final_metrics['feature_mse'][i]
    status = "✅" if mse < 0.1 else "⚠️" if mse < 0.5 else "❌"
    print(f"  {name:15}: {mse:.6f} {status}")

# Inverse scaling analysis
with open(os.path.join(DATA_DIR, "scalers.pkl"), "rb") as f:
    scalers = pickle.load(f)

# Test inverse scaling on one sequence
test_seq = test[:1]
h_test = embedder(test_seq, training=False)
x_rec_test = recovery(h_test, training=False).numpy()

# Apply inverse scaling
n, T, D = x_rec_test.shape
x_rec_inv = np.empty_like(x_rec_test)
x_orig_inv = np.empty_like(test_seq)

for d in range(D):
    if isinstance(scalers, list):
        x_rec_inv[:, :, d] = scalers[d].inverse_transform(x_rec_test[:, :, d].reshape(-1, 1)).reshape(T)
        x_orig_inv[:, :, d] = scalers[d].inverse_transform(test_seq[:, :, d].reshape(-1, 1)).reshape(T)
    else:
        x_rec_inv = scalers.inverse_transform(x_rec_test.reshape(-1, D)).reshape(n, T, D)
        x_orig_inv = scalers.inverse_transform(test_seq.reshape(-1, D)).reshape(n, T, D)
        break

# Calculate inverse-scaled errors
inverse_errors = []
for i in range(D):
    orig_feat = x_orig_inv[0, :, i]
    rec_feat = x_rec_inv[0, :, i]
    mae = np.mean(np.abs(orig_feat - rec_feat))
    inverse_errors.append(mae)

print(f"\n📈 Inverse-scaled MAE (first sequence):")
for i, name in enumerate(feature_names):
    error = inverse_errors[i]
    if name in ['Open', 'High', 'Low', 'Close']:
        avg_price = np.mean(np.abs(x_orig_inv[0, :, i]))
        rel_error = (error / avg_price) * 100 if avg_price > 0 else 0
        status = "✅" if rel_error < 10 else "⚠️" if rel_error < 30 else "❌"
        print(f"  {name:15}: {error:.2f} ({rel_error:.1f}%) {status}")
    else:
        status = "✅" if error < 1.0 else "⚠️" if error < 5.0 else "❌"
        print(f"  {name:15}: {error:.4f} {status}")

# Save comprehensive training report
training_report = {
    'training_summary': {
        'final_epoch': epoch,
        'best_val_loss': best_val_loss,
        'early_stopped': wait >= PATIENCE,
        'total_epochs_trained': epoch
    },
    'final_metrics': final_metrics,
    'inverse_scaled_errors': {name: float(inverse_errors[i]) for i, name in enumerate(feature_names)},
    'model_architecture': {
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'feature_dim': FEATURE_DIM,
        'seq_len': SEQ_LEN
    },
    'training_parameters': {
        'learning_rate': LR,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'patience': PATIENCE
    }
}

with open(os.path.join(CKPT_DIR, "enhanced_training_report.json"), "w") as f:
    json.dump(training_report, f, indent=2)

# Save reconstruction examples
np.save(os.path.join(CKPT_DIR, "enhanced_recon_examples_orig.npy"), x_orig_inv)
np.save(os.path.join(CKPT_DIR, "enhanced_recon_examples_rec.npy"), x_rec_inv)

print(f"\n🎉 ENHANCED RECONSTRUCTION TRAINING COMPLETED!")
print(f"📁 Checkpoint directory: {CKPT_DIR}")
print(f"📊 TensorBoard logs: {train_log_dir}")
print(f"📈 Best validation loss: {best_val_loss:.6f}")

# Final quality assessment
good_price_features = 0
for i, name in enumerate(feature_names[:4]):
    if name in ['Open', 'High', 'Low', 'Close']:
        avg_price = np.mean(np.abs(x_orig_inv[0, :, i]))
        if avg_price > 0:
            rel_error = inverse_errors[i] / avg_price
            if rel_error < 0.3:  # 30% relative error threshold
                good_price_features += 1

total_price_features = 4

if good_price_features >= 3:
    print("🎉 EXCELLENT: Price feature reconstruction is good! Proceed to TimeGAN.")
elif good_price_features >= 2:
    print("✅ ACCEPTABLE: Price feature reconstruction is acceptable. Proceed to TimeGAN.")
else:
    print("❌ POOR: Price feature reconstruction needs improvement. Check data preprocessing.")