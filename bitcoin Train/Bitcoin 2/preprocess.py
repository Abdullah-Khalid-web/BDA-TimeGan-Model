# -------------------------------------------------------
# Bitcoin Preprocessing with Fixed 50K Window Sampling
# -------------------------------------------------------

import argparse, os, pickle, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------

MAX_WINDOWS = 50_000


def preprocess_bitcoin(input_path, out_dir, seq_len=168, stride=1, smooth=True):

    os.makedirs(out_dir, exist_ok=True)

    print("\n=== BITCOIN PREPROCESSING (50K CAP MODE) ===")

    # ---------------------------------------------------
    # LOAD FILE
    # ---------------------------------------------------

    print("\nLoading BTC data...")

    df = pd.read_csv(
        input_path,
        sep="\t",
        na_values="?|nan|NaN",
        engine="python"
    )

    print("Rows loaded:", len(df))

    # ---------------------------------------------------
    # TIMESTAMP
    # ---------------------------------------------------

    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
    df = df.drop(columns=["Timestamp"])
    df = df.set_index("Datetime").sort_index()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(method="linear", limit_direction="both")

    # ---------------------------------------------------
    # FEATURES
    # ---------------------------------------------------

    if "Close" in df.columns:
        df["Price_Change"] = df["Close"].pct_change()
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    if "Volume" in df.columns:
        df["Volume_MA"] = df["Volume"].rolling(60).mean()
        df["Volume_Spike"] = (df["Volume"] / df["Volume_MA"]).clip(upper=10)

    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Ratio"] = df["High"] / df["Low"]

    df["Volatility"] = df["Price_Change"].rolling(60).std()

    # TIME
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Optional smoothing
    if smooth:
        df = df.rolling(5, min_periods=1).mean()

    # Extreme clip
    for col in ["Price_Change", "Log_Return", "Volatility", "Volume"]:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    df = df.dropna()

    print("Final rows used:", len(df))
    print("Total features:", df.shape[1])
    
    # ---------------------------------------------------
    # WINDOW CREATION (ONLINE SAMPLING)
    # ---------------------------------------------------

    print("\nSampling windows...")

    arr = df.values.astype(np.float32)
    N, D = arr.shape

    rng = np.random.default_rng(42)

    max_start = N - seq_len

    if max_start <= 0:
        raise RuntimeError("Dataset smaller than sequence length!")

    all_starts = np.arange(0, max_start, stride)

    print("Available windows:", len(all_starts))

    # RANDOMLY SELECT 50K WINDOWS
    if len(all_starts) > MAX_WINDOWS:
        selected = rng.choice(all_starts, size=MAX_WINDOWS, replace=False)
    else:
        selected = all_starts

    selected = np.sort(selected)

    print("Selected windows:", len(selected))

    windows = np.empty((len(selected), seq_len, D), dtype=np.float32)

    for i, s in enumerate(selected):
        windows[i] = arr[s : s+seq_len]

    # ---------------------------------------------------
    # SPLIT
    # ---------------------------------------------------

    rng.shuffle(windows)

    n = len(windows)

    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train = windows[:n_train]
    val   = windows[n_train:n_train+n_val]
    test  = windows[n_train+n_val:]

    # ---------------------------------------------------
    # SCALING
    # ---------------------------------------------------

    scalers = []

    train_s = np.zeros_like(train)
    val_s   = np.zeros_like(val)
    test_s  = np.zeros_like(test)

    for d in range(D):
        sc = StandardScaler()
        sc.fit(train[:, :, d].reshape(-1, 1))

        train_s[:,:,d] = sc.transform(train[:,:,d].reshape(-1,1)).reshape(train[:,:,d].shape)
        val_s[:,:,d] = sc.transform(val[:,:,d].reshape(-1,1)).reshape(val[:,:,d].shape)
        test_s[:,:,d] = sc.transform(test[:,:,d].reshape(-1,1)).reshape(test[:,:,d].shape)

        scalers.append(sc)

    # ---------------------------------------------------
    # SAVE
    # ---------------------------------------------------

    np.save(out_dir + "/train.npy", train_s)
    np.save(out_dir + "/val.npy", val_s)
    np.save(out_dir + "/test.npy", test_s)

    with open(out_dir + "/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    with open(out_dir + "/features.txt", "w") as f:
        f.write("\n".join(df.columns.tolist()))

    meta = {
        "MAX_WINDOWS": MAX_WINDOWS,
        "windows_used": len(windows),
        "seq_len": seq_len,
        "stride": stride,
        "train": len(train),
        "val": len(val),
        "test": len(test)
    }

    with open(out_dir + "/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ PREPROCESS COMPLETE")
    print("Saved windows:", len(windows))
    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))


# -------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no_smooth", action="store_true")

    args = parser.parse_args()

    preprocess_bitcoin(
        args.input,
        args.out_dir,
        args.seq_len,
        args.stride,
        smooth=not args.no_smooth
    )
