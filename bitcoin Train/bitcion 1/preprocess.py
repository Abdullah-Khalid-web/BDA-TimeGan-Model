import argparse, os, pickle, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_bitcoin_csv(input_path, out_dir, seq_len=168, stride=24, smooth=True):
    os.makedirs(out_dir, exist_ok=True)

    print("🔄 ENHANCED BITCOIN PREPROCESSING (CSV)")
    print("=" * 50)

    # Read CSV data
    print("Reading CSV data...")
    try:
        # Try reading with different separators
        df = pd.read_csv(input_path, sep=',', na_values="?", low_memory=False)
        if df.shape[1] == 1:  # If only one column, try semicolon
            df = pd.read_csv(input_path, sep=';', na_values="?", low_memory=False)
        if df.shape[1] == 1:  # If still one column, try tab
            df = pd.read_csv(input_path, sep='\t', na_values="?", low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    print("Columns:", df.columns.tolist())
    print("Data shape:", df.shape)
    print("First few rows:")
    print(df.head())

    # Handle different timestamp column names
    timestamp_col = None
    for col in df.columns:
        if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("❌ No timestamp column found. Available columns:", df.columns.tolist())
        return False

    print(f"Using timestamp column: {timestamp_col}")

    # Parse datetime - handle both Unix timestamp and string formats
    try:
        # Try parsing as Unix timestamp first
        df["Datetime"] = pd.to_datetime(df[timestamp_col], unit='s')
    except:
        try:
            # Try parsing as string datetime
            df["Datetime"] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            print(f"❌ Error parsing datetime: {e}")
            return False

    df = df.drop(columns=[timestamp_col])
    df = df.set_index("Datetime").sort_index()

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Remove rows with all NaN
    df = df.dropna(how='all')
    
    print(f"After initial cleaning: {df.shape}")

    # Check time frequency and resample if needed
    time_diffs = df.index.to_series().diff().dropna()
    if len(time_diffs) > 0:
        most_common_interval = time_diffs.mode().iloc[0]
        print(f"Most common time interval: {most_common_interval}")
        
        if most_common_interval != pd.Timedelta('1 hour'):
            print("Resampling to hourly frequency...")
            df = df.resample("H").mean()
    else:
        print("⚠️  No time differences found")

    # Enhanced interpolation
    df = df.interpolate(method='linear', limit_direction="both")
    df = df.ffill().bfill()  # Additional fill for safety

    print("After cleaning and resampling:", df.shape)

    # --- ENHANCED BITCOIN FEATURE ENGINEERING ---
    print("🔧 Enhanced feature engineering...")
    
    # Standardize column names (case-insensitive)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower:
            column_mapping[col] = 'High'
        elif 'low' in col_lower:
            column_mapping[col] = 'Low'
        elif 'close' in col_lower:
            column_mapping[col] = 'Close'
        elif 'volume' in col_lower:
            column_mapping[col] = 'Volume'

    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Renamed columns: {column_mapping}")

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        # Continue with available columns
        available_cols = [col for col in required_cols if col in df.columns]
        print(f"Using available columns: {available_cols}")
        if not available_cols:
            print("❌ No required columns available!")
            return False
    else:
        available_cols = required_cols

    # Price-based features
    if 'Close' in df.columns:
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=24, min_periods=1).std()
    
    if 'Volume' in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=24, min_periods=1).mean()
    
    # Enhanced outlier handling with robust clipping
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q1, upper=q99)
    
    # Volume and volatility features with conservative clipping
    vol_cols = ['Volume', 'Price_Change', 'Volatility']
    for col in vol_cols:
        if col in df.columns:
            q1, q99 = df[col].quantile([0.02, 0.98])  # More conservative
            df[col] = df[col].clip(lower=q1, upper=q99)
    
    # Additional robust features
    if all(col in df.columns for col in ['High', 'Low']):
        df['High_Low_Ratio'] = (df['High'] / df['Low']).clip(upper=1.02, lower=0.98)
    
    if all(col in df.columns for col in ['Volume', 'Volume_MA']):
        df['Volume_Spike'] = (df['Volume'] / df['Volume_MA']).clip(upper=3.0, lower=0.33)
    
    # Time-based features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(float)
    
    # Enhanced returns with clipping
    if 'Close' in df.columns:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        # Clip extreme returns
        q1, q99 = df['Log_Return'].quantile([0.01, 0.99])
        df['Log_Return'] = df['Log_Return'].clip(lower=q1, upper=q99)

    # Light smoothing if enabled
    if smooth:
        smooth_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in smooth_cols:
            if col in df.columns:
                df[col] = df[col].rolling(window=3, min_periods=1, center=True).mean()

    # Final cleanup
    df = df.dropna()
    print("After feature engineering:", df.shape)

    data = df.values.astype(np.float32)
    feature_names = df.columns.tolist()
    print("📊 Final features:", feature_names)

    # --- ENHANCED WINDOWING ---
    def make_windows(arr, seq_len, stride):
        N, D = arr.shape
        windows = []
        for start in range(0, N - seq_len + 1, stride):
            windows.append(arr[start:start + seq_len])
        return np.stack(windows)

    windows = make_windows(data, seq_len, stride)
    print("Windowed shape:", windows.shape)

    if windows.shape[0] == 0:
        print("❌ No windows created! Check sequence length and data size.")
        return False

    # --- ENHANCED SPLITTING WITH BETTER SHUFFLE ---
    np.random.seed(42)
    indices = np.random.permutation(windows.shape[0])
    windows = windows[indices]

    n = windows.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train, val, test = (
        windows[:n_train],
        windows[n_train:n_train + n_val],
        windows[n_train + n_val:]
    )

    print(f"Dataset split - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    # --- ENHANCED SCALING WITH BETTER DISTRIBUTION PRESERVATION ---
    print("📈 Applying enhanced scaling...")
    
    D = train.shape[2]
    scalers = []
    train_s = np.empty_like(train)
    val_s = np.empty_like(val)
    test_s = np.empty_like(test)

    for d in range(D):
        # Use RobustScaler for better outlier handling
        s = RobustScaler(quantile_range=(5, 95))  # Use 5th-95th percentiles
        
        # Fit on training data
        train_flat = train[:, :, d].reshape(-1, 1)
        s.fit(train_flat)
        scalers.append(s)

        # Transform all datasets
        train_s[:, :, d] = s.transform(train_flat).reshape(train[:, :, d].shape)
        val_s[:, :, d] = s.transform(val[:, :, d].reshape(-1, 1)).reshape(val_s[:, :, d].shape)
        test_s[:, :, d] = s.transform(test[:, :, d].reshape(-1, 1)).reshape(test_s[:, :, d].shape)

    # --- ENHANCED DATA VALIDATION ---
    print("\n🔍 DATA VALIDATION REPORT:")
    print("-" * 40)
    
    # Check for NaN/Inf
    print(f"NaN in train: {np.isnan(train_s).sum()}")
    print(f"Inf in train: {np.isinf(train_s).sum()}")
    
    # Check scaling results
    print("Scaled data statistics:")
    for i, name in enumerate(feature_names):
        train_feat = train_s[:, :, i].flatten()
        print(f"{name:15}: mean={train_feat.mean():7.4f}, std={train_feat.std():7.4f}, "
              f"range=[{train_feat.min():7.4f}, {train_feat.max():7.4f}]")

    # --- SAVE ENHANCED DATA ---
    print("\n💾 Saving enhanced processed data...")
    
    np.save(os.path.join(out_dir, "train.npy"), train_s)
    np.save(os.path.join(out_dir, "val.npy"), val_s)
    np.save(os.path.join(out_dir, "test.npy"), test_s)

    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))

    # Enhanced metadata
    meta = {
        "seq_len": seq_len,
        "stride": stride,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
        "features": feature_names,
        "smooth": smooth,
        "data_type": "bitcoin_enhanced",
        "scaling": "RobustScaler(5-95)",
        "total_sequences": n,
        "input_file": input_path,
        "feature_stats": {
            name: {
                "mean": float(train_s[:, :, i].mean()),
                "std": float(train_s[:, :, i].std()),
                "min": float(train_s[:, :, i].min()),
                "max": float(train_s[:, :, i].max())
            } for i, name in enumerate(feature_names)
        }
    }
    
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Enhanced preprocessing completed!")
    print(f"📁 Output directory: {out_dir}")
    print(f"📊 Final shapes - Train: {train_s.shape}, Val: {val_s.shape}, Test: {test_s.shape}")
    print(f"📈 Total sequences: {n}")
    
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Enhanced Bitcoin Data Preprocessing for CSV")
    p.add_argument("--input", required=True, help="Path to bitcoin CSV data file")
    p.add_argument("--out_dir", required=True, help="Output directory for processed data")
    p.add_argument("--seq_len", type=int, default=168, help="Sequence length for time series windows")
    p.add_argument("--stride", type=int, default=24, help="Stride for window creation")
    p.add_argument("--no_smooth", action="store_true", help="Disable rolling mean smoothing")
    
    args = p.parse_args()

    print(f"🔄 Processing: {args.input}")
    print(f"📁 Output: {args.out_dir}")
    print(f"⚙️  Settings: seq_len={args.seq_len}, stride={args.stride}, smooth={not args.no_smooth}")

    success = preprocess_bitcoin_csv(
        args.input,
        args.out_dir,
        args.seq_len,
        args.stride,
        smooth=not args.no_smooth
    )
    
    if success:
        print("🎉 Preprocessing completed successfully!")
    else:
        print("❌ Preprocessing failed!")
        exit(1)