import argparse, os, pickle, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_stock_prices(input_path, out_dir, seq_len=168, stride=12, smooth=True):
    os.makedirs(out_dir, exist_ok=True)

    print("Reading raw data...")
    df = pd.read_csv(input_path)

    # --- Parse datetime ---
    df["Datetime"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.drop(columns=["date"])
    df = df.set_index("Datetime").sort_index()

    # Select only numeric features (drop Name column)
    feature_cols = ["open", "high", "low", "close", "volume"]
    
    print("Initial shape:", df.shape)
    print("Unique stocks:", df["Name"].nunique())

    # Process each stock separately and concatenate
    all_windows = []
    
    for stock_name in df["Name"].unique():
        stock_df = df[df["Name"] == stock_name][feature_cols].copy()
        stock_df = stock_df.sort_index()
        
        # Convert all columns to numeric
        stock_df = stock_df.apply(pd.to_numeric, errors="coerce")
        
        # ⭐ LOG-TRANSFORM VOLUME to handle extreme outliers
        # Add 1 to avoid log(0), then take natural log
        stock_df["volume"] = np.log1p(stock_df["volume"])
        
        # Interpolate missing values
        stock_df = stock_df.interpolate(limit_direction="both")
        
        if smooth:
            # Light rolling mean to smooth spikes
            stock_df = stock_df.rolling(window=3, min_periods=1, center=True).mean()
        
        stock_df = stock_df.dropna()
        
        # Only use stocks with enough data points
        if len(stock_df) < seq_len:
            continue
            
        stock_data = stock_df.values.astype(np.float32)
        
        # --- Windowing ---
        def make_windows(arr, seq_len, stride):
            N, D = arr.shape
            windows = []
            for start in range(0, N - seq_len + 1, stride):
                windows.append(arr[start:start + seq_len])
            return np.stack(windows) if windows else np.array([])
        
        windows = make_windows(stock_data, seq_len, stride)
        if len(windows) > 0:
            all_windows.append(windows)
    
    # Concatenate all windows from all stocks
    windows = np.concatenate(all_windows, axis=0)
    print("Windowed shape (all stocks):", windows.shape)

    feature_names = feature_cols
    print("Features:", feature_names)

    # --- Train/Val/Test split (reproducible shuffle) ---
    np.random.seed(42)
    np.random.shuffle(windows)

    n = windows.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train, val, test = (
        windows[:n_train],
        windows[n_train:n_train + n_val],
        windows[n_train + n_val:]
    )

    # --- Per-feature scaling ---
    D = train.shape[2]
    scalers = []
    train_s = np.empty_like(train)
    val_s = np.empty_like(val)
    test_s = np.empty_like(test)

    for d in range(D):
        s = StandardScaler()
        s.fit(train[:, :, d].reshape(-1, 1))
        scalers.append(s)

        train_s[:, :, d] = s.transform(train[:, :, d].reshape(-1, 1)).reshape(train[:, :, d].shape)
        val_s[:, :, d] = s.transform(val[:, :, d].reshape(-1, 1)).reshape(val_s[:, :, d].shape)
        test_s[:, :, d] = s.transform(test[:, :, d].reshape(-1, 1)).reshape(test_s[:, :, d].shape)

    # --- Save processed arrays and scalers ---
    np.save(os.path.join(out_dir, "train.npy"), train_s)
    np.save(os.path.join(out_dir, "val.npy"), val_s)
    np.save(os.path.join(out_dir, "test.npy"), test_s)

    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))

    meta = {
        "seq_len": seq_len,
        "stride": stride,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
        "features": feature_names,
        "smooth": smooth,
        "volume_transform": "log1p"  # ⭐ Note: volume is log-transformed
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved arrays and metadata to", out_dir)
    print("Train/Val/Test shapes:", train_s.shape, val_s.shape, test_s.shape)
    
    # Print feature ranges after scaling
    print("\n⭐ Feature ranges after preprocessing (should be ~similar):")
    for i, name in enumerate(feature_names):
        feat_data = train_s[:, :, i]
        print(f"  {name}: range={feat_data.max() - feat_data.min():.2f}, std={feat_data.std():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to stock_prices.csv")
    p.add_argument("--out_dir", default="data/processed/stock_prices")
    p.add_argument("--seq_len", type=int, default=168)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--no_smooth", action="store_true",
                   help="Disable rolling mean smoothing")
    args = p.parse_args()

    preprocess_stock_prices(
        args.input,
        args.out_dir,
        args.seq_len,
        args.stride,
        smooth=not args.no_smooth
    )