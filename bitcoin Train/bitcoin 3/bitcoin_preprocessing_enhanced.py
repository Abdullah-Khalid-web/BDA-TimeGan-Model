# bitcoin_preprocessing_enhanced.py
import argparse, os, pickle, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from scipy import stats

MAX_WINDOWS = 50_000

def preprocess_bitcoin_enhanced(input_path, out_dir, seq_len=168, stride=12):
    """
    Enhanced preprocessing for financial time series
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== ENHANCED BITCOIN PREPROCESSING ===")
    
    # Load data
    df = pd.read_csv(input_path, sep="\t", engine="python")
    print(f"Rows loaded: {len(df):,}")
    
    # Basic preprocessing
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
    df = df.set_index("Datetime").sort_index()
    
    # Handle missing values more robustly
    df = df.interpolate(method="time", limit_direction="both")
    df = df.ffill().bfill()
    
    # Remove periods with no trading (zero volume for too long)
    if "Volume" in df.columns:
        volume_ma = df["Volume"].rolling(60).mean()
        df = df[volume_ma > volume_ma.quantile(0.05)]
    
    # Enhanced feature engineering for financial time series
    # Price-based features
    if "Close" in df.columns:
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
        
        # Technical indicators
        df["MA_7"] = df["Close"].rolling(7).mean()
        df["MA_21"] = df["Close"].rolling(21).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        
        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # ATR for volatility
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()
        
        # Volatility measures
        df["Volatility_5"] = df["Returns"].rolling(5).std()
        df["Volatility_20"] = df["Returns"].rolling(20).std()
        df["Volatility_60"] = df["Returns"].rolling(60).std()
        
        # Momentum
        df["Momentum_5"] = df["Close"].pct_change(5)
        df["Momentum_10"] = df["Close"].pct_change(10)
        df["Momentum_20"] = df["Close"].pct_change(20)
    
    # Volume features
    if "Volume" in df.columns:
        df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
        df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]
        df["Volume_Spike"] = (df["Volume"] / df["Volume_MA_5"]).clip(upper=10)
        
        # Price-Volume correlation
        df["VWAP"] = ((df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    
    # Time features with cyclical encoding
    df["Hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["Day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["Month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Market regime indicators
    if "Close" in df.columns:
        df["Trend"] = np.where(df["Close"] > df["MA_50"], 1, -1)
        df["Volatility_Regime"] = pd.qcut(df["Volatility_20"], q=4, labels=[0, 1, 2, 3])
    
    # Drop NaN values
    df = df.dropna()
    
    # Remove extreme outliers using IQR
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ["Hour_sin", "Hour_cos", "Day_sin", "Day_cos", "Month_sin", "Month_cos", "Trend"]:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    print(f"Final rows: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    
    # Window creation with better sampling
    arr = df.values.astype(np.float32)
    N, D = arr.shape
    
    # Create overlapping windows with more diversity
    max_start = N - seq_len
    if max_start <= 0:
        raise ValueError("Data too short for sequence length")
    
    # Stratified sampling based on volatility regimes
    if "Volatility_Regime" in df.columns:
        regime_col_idx = df.columns.get_loc("Volatility_Regime")
        regimes = arr[:, regime_col_idx]
        
        # Sample proportionally from each regime
        windows = []
        regime_ids = np.unique(regimes[:max_start])
        
        for regime in regime_ids:
            regime_indices = np.where(regimes[:max_start] == regime)[0]
            if len(regime_indices) > 0:
                n_samples = min(MAX_WINDOWS // len(regime_ids), len(regime_indices))
                selected = np.random.choice(regime_indices, size=n_samples, replace=False)
                for s in selected:
                    windows.append(arr[s:s+seq_len])
        
        windows = np.array(windows)
        windows = windows[:MAX_WINDOWS]
    else:
        # Regular random sampling
        all_starts = np.arange(0, max_start, stride)
        if len(all_starts) > MAX_WINDOWS:
            selected = np.random.choice(all_starts, size=MAX_WINDOWS, replace=False)
        else:
            selected = all_starts
        
        windows = np.array([arr[s:s+seq_len] for s in selected])
    
    print(f"Created {len(windows):,} windows")
    
    # Split data
    np.random.shuffle(windows)
    n = len(windows)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    train = windows[:n_train]
    val = windows[n_train:n_train+n_val]
    test = windows[n_train+n_val:]
    
    # Enhanced scaling - different strategies for different features
    feature_names = df.columns.tolist()
    scalers = {}
    
    # Group features for different scaling strategies
    price_features = [i for i, name in enumerate(feature_names) 
                     if any(x in name.lower() for x in ['close', 'open', 'high', 'low', 'price'])]
    volume_features = [i for i, name in enumerate(feature_names) 
                      if 'volume' in name.lower()]
    return_features = [i for i, name in enumerate(feature_names) 
                      if any(x in name.lower() for x in ['return', 'momentum', 'rsi', 'macd'])]
    volatility_features = [i for i, name in enumerate(feature_names) 
                         if 'volatility' in name.lower() or 'atr' in name.lower()]
    time_features = [i for i, name in enumerate(feature_names) 
                    if any(x in name for x in ['sin', 'cos', 'hour', 'day', 'month'])]
    
    # Apply scaling
    train_scaled = np.zeros_like(train)
    val_scaled = np.zeros_like(val)
    test_scaled = np.zeros_like(test)
    
    for idx, group in enumerate([price_features, volume_features, return_features, 
                                volatility_features, time_features]):
        if group:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            
            # Reshape for scaling
            train_group = train[:, :, group].reshape(-1, len(group))
            val_group = val[:, :, group].reshape(-1, len(group))
            test_group = test[:, :, group].reshape(-1, len(group))
            
            # Fit and transform
            train_scaled_group = sc.fit_transform(train_group)
            val_scaled_group = sc.transform(val_group)
            test_scaled_group = sc.transform(test_group)
            
            # Reshape back
            train_scaled[:, :, group] = train_scaled_group.reshape(train.shape[0], 
                                                                  train.shape[1], len(group))
            val_scaled[:, :, group] = val_scaled_group.reshape(val.shape[0], 
                                                              val.shape[1], len(group))
            test_scaled[:, :, group] = test_scaled_group.reshape(test.shape[0], 
                                                                test.shape[1], len(group))
            
            scalers[f"group_{idx}"] = {
                "scaler": sc,
                "features": [feature_names[i] for i in group]
            }
    
    # Save everything
    np.save(os.path.join(out_dir, "train.npy"), train_scaled)
    np.save(os.path.join(out_dir, "val.npy"), val_scaled)
    np.save(os.path.join(out_dir, "test.npy"), test_scaled)
    
    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    
    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_names))
    
    meta = {
        "total_windows": len(windows),
        "seq_len": seq_len,
        "features": len(feature_names),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "feature_groups": {
            "price": len(price_features),
            "volume": len(volume_features),
            "returns": len(return_features),
            "volatility": len(volatility_features),
            "time": len(time_features)
        }
    }
    
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n✅ Enhanced preprocessing complete!")
    return df