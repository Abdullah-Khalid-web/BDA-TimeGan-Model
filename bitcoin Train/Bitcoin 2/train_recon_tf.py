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

# =========================================================
# PERFORMANCE SETTINGS
# =========================================================
tf.config.optimizer.set_jit(True)

# Enable mixed precision ONLY if GPU exists
if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled (GPU)")
else:
    print("⚠️ CPU detected — mixed precision disabled")

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data/processed/crypto"
CKPT_DIR = "outputs/checkpoints/timegan_optimized"
os.makedirs(CKPT_DIR, exist_ok=True)

# =========================================================
# AUTO DIM DETECTION
# =========================================================
train_sample = np.load(
    os.path.join(DATA_DIR, "train.npy"),
    mmap_mode="r"
)

SEQ_LEN     = train_sample.shape[1]
FEATURE_DIM= train_sample.shape[2]

print(f"Detected SEQ_LEN={SEQ_LEN}, FEATURES={FEATURE_DIM}")

# =========================================================
# MEMORY OPTIMIZED PARAMS
# =========================================================
HIDDEN_DIM = 64
Z_DIM = 32
BATCH_SIZE = 32

LR_E = 5e-5
LR_G = 5e-5
LR_D = 2e-5

EPOCHS_SUPERVISOR  = 20
EPOCHS_ADVERSARIAL= 100

N_CRITIC = 2

LAMBDA_SUP  = 0.1
LAMBDA_REC  = 2.0
LAMBDA_GP   = 10.0
LAMBDA_STAT = 10.0

VALIDATION_EVAL_EVERY = 5
VAL_PATIENCE = 15

# =========================================================
# TENSORBOARD
# =========================================================
log_dir = f"logs/timegan_optimized/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = tf.summary.create_file_writer(log_dir)

# =========================================================
# DATA PIPELINE (STREAMING)
# =========================================================
train = np.load(os.path.join(DATA_DIR,"train.npy"), mmap_mode="r")
val   = np.load(os.path.join(DATA_DIR,"val.npy"), mmap_mode="r")

train_ds = (
    tf.data.Dataset.from_tensor_slices(train)
    .shuffle(5000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# =========================================================
# MODELS
# =========================================================

embedder = Embedder(FEATURE_DIM, HIDDEN_DIM, num_layers=2)
recovery = Recovery(HIDDEN_DIM, FEATURE_DIM, num_layers=2)
generator = Generator(Z_DIM, HIDDEN_DIM, num_layers=2)
supervisor = Supervisor(HIDDEN_DIM, num_layers=1)
discriminator = Discriminator(HIDDEN_DIM, num_layers=1)

# Build
_ = embedder(tf.zeros([1,SEQ_LEN,FEATURE_DIM]))
_ = recovery(tf.zeros([1,SEQ_LEN,HIDDEN_DIM]))
_ = generator(tf.zeros([1,SEQ_LEN,Z_DIM]))
_ = supervisor(tf.zeros([1,SEQ_LEN,HIDDEN_DIM]))
_ = discriminator(tf.zeros([1,SEQ_LEN,HIDDEN_DIM]))

print("✅ Models built")

# =========================================================
# OPTIMIZERS
# =========================================================

opt_e = optimizers.Adam(LR_E, beta_1=0.5)
opt_g = optimizers.Adam(LR_G, beta_1=0.5)
opt_s = optimizers.Adam(LR_G, beta_1=0.5)
opt_d = optimizers.Adam(LR_D, beta_1=0.5)

# =========================================================
# CHECKPOINT
# =========================================================

ckpt = tf.train.Checkpoint(
    embedder=embedder,
    recovery=recovery,
    generator=generator,
    supervisor=supervisor,
    discriminator=discriminator,
    opt_e=opt_e,
    opt_g=opt_g,
    opt_s=opt_s,
    opt_d=opt_d,
)

manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=3)

# =========================================================
# TRAINING STATE
# =========================================================

state_file = os.path.join(CKPT_DIR, "state.json")

class State:
    epoch=0
    best=1e9
    wait=0
    sup_epoch=0

state = State()

if manager.latest_checkpoint and os.path.exists(state_file):
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    with open(state_file) as f:
        s=json.load(f)

    state.epoch = s["epoch"]
    state.best  = s["best"]
    state.wait  = s["wait"]
    state.sup_epoch=s["sup"]

    print("✅ Resuming training")

# =========================================================
# UTILITIES
# =========================================================

def save_state():
    with open(state_file,"w") as f:
        json.dump({
            "epoch":state.epoch,
            "best":state.best,
            "wait":state.wait,
            "sup":state.sup_epoch
        },f)

def sample_z(bs):
    return tf.random.normal([bs,SEQ_LEN,Z_DIM], stddev=0.5)

# =========================================================
# TRAIN STEPS
# =========================================================

@tf.function
def train_supervised(x):
    with tf.GradientTape() as t:
        H = embedder(x)
        H_hat = supervisor(H)
        loss = tf.reduce_mean(tf.keras.losses.MSE(H[:,1:], H_hat[:,:-1]))

    g=t.gradient(loss, supervisor.trainable_variables)
    opt_s.apply_gradients(zip(g, supervisor.trainable_variables))

    return loss


@tf.function
def critic_step(x):
    bs=tf.shape(x)[0]
    z=sample_z(bs)

    with tf.GradientTape() as t:
        Hr = embedder(x)
        Ef = generator(z)
        Hf = supervisor(Ef)

        Dr = discriminator(Hr)
        Df = discriminator(Hf)

        adv = tf.reduce_mean(Df)-tf.reduce_mean(Dr)

        gp = tf.reduce_mean(tf.square(
            tf.norm(
                tf.random.uniform([bs,1,1]) * Hr +
                (1-tf.random.uniform([bs,1,1])) * Hf
            ,axis=[1,2]) - 1.0
        ))

        loss = adv + LAMBDA_GP*gp

    g=t.gradient(loss, discriminator.trainable_variables)
    opt_d.apply_gradients(zip(g,discriminator.trainable_variables))

    return adv,gp


@tf.function
def generator_step(x):
    bs=tf.shape(x)[0]
    z=sample_z(bs)

    with tf.GradientTape() as t:
        Ef=generator(z)
        Hf=supervisor(Ef)
        Xf=recovery(Hf)

        Df=discriminator(Hf)

        Hr=embedder(x)

        adv = -tf.reduce_mean(Df)
        sup = tf.reduce_mean(tf.keras.losses.MSE(Hr[:,1:], Hf[:,:-1]))
        rec = tf.reduce_mean(tf.keras.losses.MSE(x, Xf))

        rm=tf.reduce_mean(x,[0,1])
        sm=tf.reduce_mean(Xf,[0,1])
        stat=tf.reduce_mean(tf.square(rm-sm))

        loss=adv + LAMBDA_SUP*sup + LAMBDA_REC*rec + LAMBDA_STAT*stat

    v=generator.trainable_variables+supervisor.trainable_variables
    g=t.gradient(loss,v)
    opt_g.apply_gradients(zip(g,v))

    return loss

# =========================================================
# VALIDATION
# =========================================================

def quick_validate():
    sample = val[:32]
    z=sample_z(32)

    Ef=generator(z)
    Hf=supervisor(Ef)
    Xf=recovery(Hf)

    rm=tf.reduce_mean(sample,[0,1])
    sm=tf.reduce_mean(Xf,[0,1])

    return float(tf.reduce_mean(tf.abs(rm-sm)))

# =========================================================
# TRAIN LOOP
# =========================================================

def run_training():

    # Supervisor warmup
    if state.sup_epoch < EPOCHS_SUPERVISOR:
        print("Supervisor pretraining")

        for e in range(state.sup_epoch+1, EPOCHS_SUPERVISOR+1):
            losses=[]

            for b in train_ds:
                losses.append(train_supervised(b).numpy())

            print(f"[SUP {e}/{EPOCHS_SUPERVISOR}] loss={np.mean(losses):.6f}")

            state.sup_epoch=e
            save_state()

    print("Adversarial training start")

    for e in range(state.epoch+1, EPOCHS_ADVERSARIAL+1):

        gl=[]

        for b in train_ds:
            for _ in range(N_CRITIC):
                critic_step(b)

            gl.append(generator_step(b).numpy())

        # Validation
        if e%VALIDATION_EVAL_EVERY==0:
            v = quick_validate()
            print(f"VAL mean diff:{v:.6f}")

            if v<state.best:
                state.best=v
                state.wait=0
                manager.save()
            else:
                state.wait+=1

            if state.wait>=VAL_PATIENCE:
                print("Early stopping")
                break

        state.epoch=e
        save_state()

        print(
            f"[GAN {e}/{EPOCHS_ADVERSARIAL}] "
            f"Loss={np.mean(gl):.6f}"
        )

# =========================================================
# GENERATE
# =========================================================

def generate(n=10):

    out=[]

    for i in range(0,n,BATCH_SIZE):
        bs=min(BATCH_SIZE,n-i)

        z=sample_z(bs)
        x=recovery(supervisor(generator(z))).numpy()

        out.append(x)

    return np.concatenate(out,0)


# =========================================================
# MAIN
# =========================================================

if __name__=='__main__':

    print("Starting optimized TimeGAN training...")
    run_training()

    embedder.save_weights(os.path.join(CKPT_DIR,"embedder_final.weights.h5"))
    recovery.save_weights(os.path.join(CKPT_DIR,"recovery_final.weights.h5"))
    generator.save_weights(os.path.join(CKPT_DIR,"generator_final.weights.h5"))
    supervisor.save_weights(os.path.join(CKPT_DIR,"supervisor_final.weights.h5"))
    discriminator.save_weights(os.path.join(CKPT_DIR,"discriminator_final.weights.h5"))

    print("✅ Weights saved")

    # Test sample
    s=generate(10)
    print("✅ Synthetic data shape:",s.shape)
    print("TensorBoard logs:",log_dir)
