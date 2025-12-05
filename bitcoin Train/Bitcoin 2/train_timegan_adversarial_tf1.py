# import os
# import time
# import pickle
# import json
# import datetime
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import optimizers
# import sys

# sys.path.append("..")
# from timegan_tf import (
#     Embedder, Recovery, Generator,
#     Supervisor, Discriminator
# )

# # =========================================================
# # PERFORMANCE SETTINGS
# # =========================================================
# tf.config.optimizer.set_jit(True)

# # Enable mixed precision ONLY if GPU exists
# if tf.config.list_physical_devices("GPU"):
#     from tensorflow.keras.mixed_precision import set_global_policy
#     set_global_policy("mixed_float16")
#     print("✅ Mixed precision enabled (GPU)")
# else:
#     print("⚠️ CPU detected — mixed precision disabled")

# # =========================================================
# # CONFIG
# # =========================================================
# DATA_DIR = "data/processed/crypto"
# CKPT_DIR = "outputs/checkpoints/timegan_optimized"
# os.makedirs(CKPT_DIR, exist_ok=True)

# # =========================================================
# # AUTO DIM DETECTION
# # =========================================================
# train_sample = np.load(
#     os.path.join(DATA_DIR, "train.npy"),
#     mmap_mode="r"
# )

# SEQ_LEN     = train_sample.shape[1]
# FEATURE_DIM= train_sample.shape[2]

# print(f"Detected SEQ_LEN={SEQ_LEN}, FEATURES={FEATURE_DIM}")

# # =========================================================
# # MEMORY OPTIMIZED PARAMS
# # =========================================================
# HIDDEN_DIM = 64
# Z_DIM = 32
# BATCH_SIZE = 32

# LR_E = 5e-5
# LR_G = 5e-5
# LR_D = 2e-5

# EPOCHS_SUPERVISOR  = 20
# EPOCHS_ADVERSARIAL= 100

# N_CRITIC = 2

# LAMBDA_SUP  = 0.1
# LAMBDA_REC  = 2.0
# LAMBDA_GP   = 10.0
# LAMBDA_STAT = 10.0

# VALIDATION_EVAL_EVERY = 5
# VAL_PATIENCE = 15

# # =========================================================
# # TENSORBOARD
# # =========================================================
# log_dir = f"logs/timegan_optimized/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
# writer = tf.summary.create_file_writer(log_dir)

# # =========================================================
# # DATA PIPELINE (STREAMING)
# # =========================================================
# train = np.load(os.path.join(DATA_DIR,"train.npy"), mmap_mode="r")
# val   = np.load(os.path.join(DATA_DIR,"val.npy"), mmap_mode="r")

# train_ds = (
#     tf.data.Dataset.from_tensor_slices(train)
#     .shuffle(5000)
#     .batch(BATCH_SIZE, drop_remainder=True)
#     .prefetch(tf.data.AUTOTUNE)
# )

# # =========================================================
# # MODELS
# # =========================================================

# embedder = Embedder(FEATURE_DIM, HIDDEN_DIM, num_layers=2)
# recovery = Recovery(HIDDEN_DIM, FEATURE_DIM, num_layers=2)
# generator = Generator(Z_DIM, HIDDEN_DIM, num_layers=2)
# supervisor = Supervisor(HIDDEN_DIM, num_layers=1)
# discriminator = Discriminator(HIDDEN_DIM, num_layers=1)

# # Build
# _ = embedder(tf.zeros([1,SEQ_LEN,FEATURE_DIM]))
# _ = recovery(tf.zeros([1,SEQ_LEN,HIDDEN_DIM]))
# _ = generator(tf.zeros([1,SEQ_LEN,Z_DIM]))
# _ = supervisor(tf.zeros([1,SEQ_LEN,HIDDEN_DIM]))
# _ = discriminator(tf.zeros([1,SEQ_LEN,HIDDEN_DIM]))

# print("✅ Models built")

# # =========================================================
# # OPTIMIZERS
# # =========================================================

# opt_e = optimizers.Adam(LR_E, beta_1=0.5)
# opt_g = optimizers.Adam(LR_G, beta_1=0.5)
# opt_s = optimizers.Adam(LR_G, beta_1=0.5)
# opt_d = optimizers.Adam(LR_D, beta_1=0.5)

# # =========================================================
# # CHECKPOINT
# # =========================================================

# ckpt = tf.train.Checkpoint(
#     embedder=embedder,
#     recovery=recovery,
#     generator=generator,
#     supervisor=supervisor,
#     discriminator=discriminator,
#     opt_e=opt_e,
#     opt_g=opt_g,
#     opt_s=opt_s,
#     opt_d=opt_d,
# )

# manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=3)

# # =========================================================
# # TRAINING STATE
# # =========================================================

# state_file = os.path.join(CKPT_DIR, "state.json")

# class State:
#     epoch=0
#     best=1e9
#     wait=0
#     sup_epoch=0

# state = State()

# if manager.latest_checkpoint and os.path.exists(state_file):
#     ckpt.restore(manager.latest_checkpoint).expect_partial()

#     with open(state_file) as f:
#         s=json.load(f)

#     state.epoch = s["epoch"]
#     state.best  = s["best"]
#     state.wait  = s["wait"]
#     state.sup_epoch=s["sup"]

#     print("✅ Resuming training")

# # =========================================================
# # UTILITIES
# # =========================================================

# def save_state():
#     with open(state_file,"w") as f:
#         json.dump({
#             "epoch":state.epoch,
#             "best":state.best,
#             "wait":state.wait,
#             "sup":state.sup_epoch
#         },f)

# def sample_z(bs):
#     return tf.random.normal([bs,SEQ_LEN,Z_DIM], stddev=0.5)

# # =========================================================
# # TRAIN STEPS
# # =========================================================

# @tf.function
# def train_supervised(x):
#     with tf.GradientTape() as t:
#         H = embedder(x)
#         H_hat = supervisor(H)
#         loss = tf.reduce_mean(tf.keras.losses.MSE(H[:,1:], H_hat[:,:-1]))

#     g=t.gradient(loss, supervisor.trainable_variables)
#     opt_s.apply_gradients(zip(g, supervisor.trainable_variables))

#     return loss


# @tf.function
# def critic_step(x):
#     bs=tf.shape(x)[0]
#     z=sample_z(bs)

#     with tf.GradientTape() as t:
#         Hr = embedder(x)
#         Ef = generator(z)
#         Hf = supervisor(Ef)

#         Dr = discriminator(Hr)
#         Df = discriminator(Hf)

#         adv = tf.reduce_mean(Df)-tf.reduce_mean(Dr)

#         gp = tf.reduce_mean(tf.square(
#             tf.norm(
#                 tf.random.uniform([bs,1,1]) * Hr +
#                 (1-tf.random.uniform([bs,1,1])) * Hf
#             ,axis=[1,2]) - 1.0
#         ))

#         loss = adv + LAMBDA_GP*gp

#     g=t.gradient(loss, discriminator.trainable_variables)
#     opt_d.apply_gradients(zip(g,discriminator.trainable_variables))

#     return adv,gp


# @tf.function
# def generator_step(x):
#     bs=tf.shape(x)[0]
#     z=sample_z(bs)

#     with tf.GradientTape() as t:
#         Ef=generator(z)
#         Hf=supervisor(Ef)
#         Xf=recovery(Hf)

#         Df=discriminator(Hf)

#         Hr=embedder(x)

#         adv = -tf.reduce_mean(Df)
#         sup = tf.reduce_mean(tf.keras.losses.MSE(Hr[:,1:], Hf[:,:-1]))
#         rec = tf.reduce_mean(tf.keras.losses.MSE(x, Xf))

#         rm=tf.reduce_mean(x,[0,1])
#         sm=tf.reduce_mean(Xf,[0,1])
#         stat=tf.reduce_mean(tf.square(rm-sm))

#         loss=adv + LAMBDA_SUP*sup + LAMBDA_REC*rec + LAMBDA_STAT*stat

#     v=generator.trainable_variables+supervisor.trainable_variables
#     g=t.gradient(loss,v)
#     opt_g.apply_gradients(zip(g,v))

#     return loss

# # =========================================================
# # VALIDATION
# # =========================================================

# def quick_validate():
#     sample = val[:32]
#     z=sample_z(32)

#     Ef=generator(z)
#     Hf=supervisor(Ef)
#     Xf=recovery(Hf)

#     rm=tf.reduce_mean(sample,[0,1])
#     sm=tf.reduce_mean(Xf,[0,1])

#     return float(tf.reduce_mean(tf.abs(rm-sm)))

# # =========================================================
# # TRAIN LOOP
# # =========================================================

# def run_training():

#     # Supervisor warmup
#     if state.sup_epoch < EPOCHS_SUPERVISOR:
#         print("Supervisor pretraining")

#         for e in range(state.sup_epoch+1, EPOCHS_SUPERVISOR+1):
#             losses=[]

#             for b in train_ds:
#                 losses.append(train_supervised(b).numpy())

#             print(f"[SUP {e}/{EPOCHS_SUPERVISOR}] loss={np.mean(losses):.6f}")

#             state.sup_epoch=e
#             save_state()

#     print("Adversarial training start")

#     for e in range(state.epoch+1, EPOCHS_ADVERSARIAL+1):

#         gl=[]

#         for b in train_ds:
#             for _ in range(N_CRITIC):
#                 critic_step(b)

#             gl.append(generator_step(b).numpy())

#         # Validation
#         if e%VALIDATION_EVAL_EVERY==0:
#             v = quick_validate()
#             print(f"VAL mean diff:{v:.6f}")

#             if v<state.best:
#                 state.best=v
#                 state.wait=0
#                 manager.save()
#             else:
#                 state.wait+=1

#             if state.wait>=VAL_PATIENCE:
#                 print("Early stopping")
#                 break

#         state.epoch=e
#         save_state()

#         print(
#             f"[GAN {e}/{EPOCHS_ADVERSARIAL}] "
#             f"Loss={np.mean(gl):.6f}"
#         )

# # =========================================================
# # GENERATE
# # =========================================================

# def generate(n=10):

#     out=[]

#     for i in range(0,n,BATCH_SIZE):
#         bs=min(BATCH_SIZE,n-i)

#         z=sample_z(bs)
#         x=recovery(supervisor(generator(z))).numpy()

#         out.append(x)

#     return np.concatenate(out,0)


# # =========================================================
# # MAIN
# # =========================================================

# if __name__=='__main__':

#     print("Starting optimized TimeGAN training...")
#     run_training()

#     embedder.save_weights(os.path.join(CKPT_DIR,"embedder_final.weights.h5"))
#     recovery.save_weights(os.path.join(CKPT_DIR,"recovery_final.weights.h5"))
#     generator.save_weights(os.path.join(CKPT_DIR,"generator_final.weights.h5"))
#     supervisor.save_weights(os.path.join(CKPT_DIR,"supervisor_final.weights.h5"))
#     discriminator.save_weights(os.path.join(CKPT_DIR,"discriminator_final.weights.h5"))

#     print("✅ Weights saved")

#     # Test sample
#     s=generate(10)
#     print("✅ Synthetic data shape:",s.shape)
#     print("TensorBoard logs:",log_dir)



























# src1/train_timegan_adversarial_tf1_stable.py
import os
import time
import pickle
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import sys

sys.path.append("..")
from timegan_tf import (
    Embedder, Recovery, Generator,
    Supervisor, Discriminator
)

# ---------- Settings ----------
tf.config.optimizer.set_jit(True)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled (GPU)")
else:
    print("⚠️ CPU detected — mixed precision disabled")

DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_optimized"
os.makedirs(CKPT_DIR, exist_ok=True)

# auto detect dims
train_sample = np.load(os.path.join(DATA_DIR, "train.npy"), mmap_mode="r")
SEQ_LEN = train_sample.shape[1]
FEATURE_DIM = train_sample.shape[2]
print(f"Detected SEQ_LEN={SEQ_LEN}, FEATURES={FEATURE_DIM}")

# ---------- Hyperparams (more conservative) ----------
HIDDEN_DIM = 64
Z_DIM = 32
BATCH_SIZE = 48              # increase if you have memory, reduces noise
LR_E = 3e-5
LR_G = 3e-5
LR_D = 8e-6                  # discriminator learning rate lower than generator
EPOCHS_SUPERVISOR = 20
EPOCHS_ADVERSARIAL = 300
N_CRITIC = 5                 # more critic steps per gen step (helps stability)
LAMBDA_GP = 10.0
LAMBDA_SUP = 0.1
LAMBDA_REC = 2.0
LAMBDA_STAT = 10.0

VALIDATION_EVAL_EVERY = 5
VAL_PATIENCE = 20

GRAD_CLIP_D = 1.0
GRAD_CLIP_G = 0.5
D_NAN_THRESHOLD = 1e5       # if discriminator loss magnitude explodes beyond this, rollback

# ---------- Logging ----------
log_dir = f"logs/timegan_optimized/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = tf.summary.create_file_writer(log_dir)

# ---------- Data pipeline ----------
train = np.load(os.path.join(DATA_DIR, "train.npy"), mmap_mode="r")
val = np.load(os.path.join(DATA_DIR, "val.npy"), mmap_mode="r")

train_ds = (
    tf.data.Dataset.from_tensor_slices(train)
    .shuffle(buffer_size=min(5000, train.shape[0]))
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# ---------- Models ----------
embedder = Embedder(FEATURE_DIM, HIDDEN_DIM, num_layers=2)
recovery = Recovery(HIDDEN_DIM, FEATURE_DIM, num_layers=2)
generator = Generator(Z_DIM, HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(HIDDEN_DIM, num_layers=1)
discriminator = Discriminator(HIDDEN_DIM, num_layers=1)

# forward build
_ = embedder(tf.zeros([1, SEQ_LEN, FEATURE_DIM]))
_ = recovery(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = generator(tf.zeros([1, SEQ_LEN, Z_DIM]))
_ = supervisor(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))
_ = discriminator(tf.zeros([1, SEQ_LEN, HIDDEN_DIM]))

# ---------- Opts ----------
opt_e = optimizers.Adam(LR_E, beta_1=0.5)
opt_g = optimizers.Adam(LR_G, beta_1=0.5)
opt_s = optimizers.Adam(LR_G, beta_1=0.5)
opt_d = optimizers.Adam(LR_D, beta_1=0.5)

# ---------- Checkpoint ----------
ckpt = tf.train.Checkpoint(
    embedder=embedder, recovery=recovery,
    generator=generator, supervisor=supervisor,
    discriminator=discriminator,
    opt_e=opt_e, opt_g=opt_g, opt_s=opt_s, opt_d=opt_d
)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)
state_file = os.path.join(CKPT_DIR, "state.json")

class State:
    epoch = 0
    sup_epoch = 0
    best_val = 1e9
    wait = 0
state = State()

if manager.latest_checkpoint and os.path.exists(state_file):
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    with open(state_file, "r") as f:
        s = json.load(f)
    state.epoch = s.get("epoch", 0)
    state.sup_epoch = s.get("sup_epoch", 0)
    state.best_val = s.get("best_val", 1e9)
    state.wait = s.get("wait", 0)
    print("✅ Resumed from checkpoint:", manager.latest_checkpoint)

def save_state():
    with open(state_file, "w") as f:
        json.dump({"epoch": state.epoch, "sup_epoch": state.sup_epoch,
                   "best_val": state.best_val, "wait": state.wait}, f)

# ---------- Utilities ----------
def sample_z(bs):
    return tf.random.normal([bs, SEQ_LEN, Z_DIM], stddev=0.5)

def add_input_noise(x, std=1e-3):
    if std <= 0.0:
        return x
    noise = tf.random.normal(tf.shape(x), stddev=std)
    return x + noise

# ---------- Stable gradient penalty ----------
def gradient_penalty(real_H, fake_H):
    batch_size = tf.shape(real_H)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    inter = alpha * real_H + (1.0 - alpha) * fake_H
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inter)
        d_inter = discriminator(inter, training=True)
    grads = gp_tape.gradient(d_inter, inter)
    grads = tf.reshape(grads, [batch_size, -1])
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return gp

# ---------- Steps ----------
@tf.function
def supervised_step(x):
    with tf.GradientTape() as tape:
        H = embedder(x, training=True)
        H_hat = supervisor(H, training=True)
        loss_s = tf.reduce_mean(tf.keras.losses.MSE(H[:,1:,:], H_hat[:,:-1,:]))
    grads = tape.gradient(loss_s, supervisor.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    opt_s.apply_gradients(zip(grads, supervisor.trainable_variables))
    return loss_s

@tf.function
def critic_step(x):
    bs = tf.shape(x)[0]
    z = sample_z(bs)
    # add tiny noise to real and fake to regularize D
    x_noisy = add_input_noise(x, std=1e-3)
    with tf.GradientTape() as tape:
        H_real = embedder(x_noisy, training=False)
        E_hat = generator(z, training=False)
        H_fake = supervisor(E_hat, training=False)
        # D outputs shape (bs, 1) or (bs, ...) depending impl; reduce_mean used below
        D_real = discriminator(H_real, training=True)
        D_fake = discriminator(H_fake, training=True)
        loss_critic = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
        gp = gradient_penalty(H_real, H_fake)
        loss = loss_critic + LAMBDA_GP * gp
    grads = tape.gradient(loss, discriminator.trainable_variables)
    grads = [tf.clip_by_norm(g, GRAD_CLIP_D) for g in grads]
    opt_d.apply_gradients(zip(grads, discriminator.trainable_variables))
    # return components for logging
    return loss_critic, gp, tf.reduce_mean(D_real), tf.reduce_mean(D_fake)

@tf.function
def generator_step(x):
    bs = tf.shape(x)[0]
    z = sample_z(bs)
    with tf.GradientTape() as tape:
        E_hat = generator(z, training=True)
        H_hat = supervisor(E_hat, training=True)
        X_hat = recovery(H_hat, training=True)
        D_fake = discriminator(H_hat, training=False)
        adv = -tf.reduce_mean(D_fake)
        H_real = embedder(x, training=False)
        sup = tf.reduce_mean(tf.keras.losses.MSE(H_real[:,1:,:], H_hat[:,:-1,:]))
        rec = tf.reduce_mean(tf.keras.losses.MSE(x, X_hat))
        rm = tf.reduce_mean(x, axis=[0,1])
        sm = tf.reduce_mean(X_hat, axis=[0,1])
        stat = tf.reduce_mean(tf.square(rm-sm))
        total_loss = adv + LAMBDA_SUP*sup + LAMBDA_REC*rec + LAMBDA_STAT*stat
    var_list = generator.trainable_variables + supervisor.trainable_variables
    grads = tape.gradient(total_loss, var_list)
    grads = [tf.clip_by_norm(g, GRAD_CLIP_G) for g in grads]
    opt_g.apply_gradients(zip(grads, var_list))
    return total_loss, adv, sup, rec, stat

# ---------- Validation ----------
def quick_validate():
    sample = val[:min(32, val.shape[0])]
    z = sample_z(tf.shape(sample)[0])
    E_hat = generator(z, training=False)
    H_hat = supervisor(E_hat, training=False)
    X_hat = recovery(H_hat, training=False)
    rm = tf.reduce_mean(sample, axis=[0,1])
    sm = tf.reduce_mean(X_hat, axis=[0,1])
    return float(tf.reduce_mean(tf.abs(rm-sm)))

# ---------- Training loop ----------
def run_training():
    # supervisor warmup
    if state.sup_epoch < EPOCHS_SUPERVISOR:
        print("Supervisor pretraining...")
        for e in range(state.sup_epoch + 1, EPOCHS_SUPERVISOR + 1):
            losses = []
            for batch in train_ds:
                losses.append(supervised_step(batch).numpy())
            state.sup_epoch = e
            save_state()
            print(f"[Sup {e}/{EPOCHS_SUPERVISOR}] loss={np.mean(losses):.6f}")

    print("Starting adversarial training...")
    for e in range(state.epoch + 1, EPOCHS_ADVERSARIAL + 1):
        d_losses = []; g_losses = []; gp_vals = []; d_real_vals = []; d_fake_vals = []
        epoch_start = time.time()
        for batch in train_ds:
            # train discriminator N_CRITIC times
            for _ in range(N_CRITIC):
                loss_crit, gp, d_real, d_fake = critic_step(batch)
                d_losses.append(float(loss_crit.numpy()))
                gp_vals.append(float(gp.numpy()))
                d_real_vals.append(float(d_real.numpy()))
                d_fake_vals.append(float(d_fake.numpy()))
            # train generator once
            g_total, g_adv, g_sup, g_rec, g_stat = generator_step(batch)
            g_losses.append(float(g_total.numpy()))
            # early detection of explosion
            if abs(g_total.numpy()) > D_NAN_THRESHOLD or any(abs(np.array(d_losses[-5:])) > D_NAN_THRESHOLD):
                print("⚠️ Loss explosion detected. Restoring last best checkpoint and reducing LR.")
                if manager.latest_checkpoint:
                    ckpt.restore(manager.latest_checkpoint).expect_partial()
                    # reduce learning rates (simple decay)
                    opt_g.learning_rate.assign(opt_g.learning_rate * 0.5)
                    opt_d.learning_rate.assign(opt_d.learning_rate * 0.5)
                    print("LR reduced. Resumed checkpoint:", manager.latest_checkpoint)
                else:
                    print("No checkpoint to rollback to.")
                break

        # validation + checkpointing
        if e % VALIDATION_EVAL_EVERY == 0:
            val_metric = quick_validate()
            print(f"Epoch {e} validation mean-diff: {val_metric:.6f}")
            with writer.as_default():
                tf.summary.scalar("val_mean_diff", val_metric, step=e)
                tf.summary.scalar("d_loss", np.mean(d_losses) if d_losses else 0.0, step=e)
                tf.summary.scalar("g_loss", np.mean(g_losses) if g_losses else 0.0, step=e)
                tf.summary.scalar("gp", np.mean(gp_vals) if gp_vals else 0.0, step=e)
                tf.summary.scalar("d_real", np.mean(d_real_vals) if d_real_vals else 0.0, step=e)
                tf.summary.scalar("d_fake", np.mean(d_fake_vals) if d_fake_vals else 0.0, step=e)
            # save best
            if val_metric < state.best_val:
                state.best_val = val_metric
                state.wait = 0
                manager.save()
                print("✅ New best val metric. Checkpoint saved.")
            else:
                state.wait += 1
                if state.wait >= VAL_PATIENCE:
                    print("⛔ Early stopping (validation patience reached).")
                    break

        state.epoch = e
        save_state()
        elapsed = time.time() - epoch_start
        print(f"[Epoch {e}/{EPOCHS_ADVERSARIAL}] D_loss={np.mean(d_losses):.4f} G_loss={np.mean(g_losses):.4f} GP={np.mean(gp_vals):.4f} time={elapsed:.1f}s")

# ---------- Generation helper ----------
def generate(n=10):
    out = []
    for i in range(0, n, BATCH_SIZE):
        bs = min(BATCH_SIZE, n - i)
        z = sample_z(bs)
        x = recovery(supervisor(generator(z, training=False), training=False), training=False).numpy()
        out.append(x)
    return np.concatenate(out, axis=0)

# ---------- Main ----------
if __name__ == "__main__":
    print("Running stable TimeGAN training...")
    run_training()
    # save final weights
    embedder.save_weights(os.path.join(CKPT_DIR, "embedder_final.weights.h5"))
    recovery.save_weights(os.path.join(CKPT_DIR, "recovery_final.weights.h5"))
    generator.save_weights(os.path.join(CKPT_DIR, "generator_final.weights.h5"))
    supervisor.save_weights(os.path.join(CKPT_DIR, "supervisor_final.weights.h5"))
    discriminator.save_weights(os.path.join(CKPT_DIR, "discriminator_final.weights.h5"))
    print("✅ Final weights saved.")
