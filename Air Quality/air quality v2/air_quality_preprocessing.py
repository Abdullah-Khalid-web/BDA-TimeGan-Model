# air_quality_preprocessing.py
import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AirQualityPreprocessor:
    def __init__(self, input_csv, output_dir, seq_len=24, stride=6):
        """
        Preprocess air quality data for TimeGAN
        
        Args:
            input_csv: Path to air1.csv
            output_dir: Directory to save processed data
            seq_len: Sequence length (hours/days)
            stride: Stride for sliding window
        """
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.seq_len = seq_len
        self.stride = stride
        self.feature_groups = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_and_clean(self):
        """Load and clean the air quality dataset"""
        print("Loading air quality data...")
        try:
            df = pd.read_csv(self.input_csv)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Try with different encoding
            df = pd.read_csv(self.input_csv, encoding='latin-1')
        
        print(f"Original shape: {df.shape}")
        
        # Check column names
        print(f"Columns: {len(df.columns)} columns")
        
        # Convert date columns
        date_columns = ['first_max_datetime', 'second_max_datetime', 
                       'third_max_datetime', 'fourth_max_datetime']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Parse year from data
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Remove rows with missing essential data
        essential_cols = ['latitude', 'longitude', 'arithmetic_mean']
        essential_cols = [col for col in essential_cols if col in df.columns]
        
        if essential_cols:
            df = df.dropna(subset=essential_cols)
        
        print(f"After cleaning: {df.shape}")
        return df
    
    def engineer_features(self, df):
        """Create engineered features for air quality"""
        print("\nEngineering features...")
        
        # 1. Geographic features
        location_cols = ['state_code', 'county_code', 'site_num']
        if all(col in df.columns for col in location_cols):
            df['location_id'] = df['state_code'].astype(str) + '_' + \
                               df['county_code'].astype(str) + '_' + \
                               df['site_num'].astype(str)
        else:
            # Create a simple index as location_id
            df['location_id'] = df.index.astype(str)
        
        # 2. Temporal features
        if 'year' in df.columns:
            if df['year'].nunique() > 1:
                df['year_normalized'] = (df['year'] - df['year'].min()) / \
                                       (df['year'].max() - df['year'].min())
            else:
                df['year_normalized'] = 0.5
        
        # 3. Pollutant type encoding
        # Create simplified pollutant category
        if 'parameter_name' in df.columns:
            df['pollutant_category'] = df['parameter_name'].astype(str).str[:20]
            
            # Create numeric encoding
            unique_pollutants = df['pollutant_category'].unique()
            pollutant_mapping = {poll: idx for idx, poll in enumerate(unique_pollutants)}
            df['pollutant_code'] = df['pollutant_category'].map(pollutant_mapping)
        else:
            df['pollutant_code'] = 0
        
        # 4. Measurement statistics features
        # Create ratio features
        if all(col in df.columns for col in ['arithmetic_mean', 'arithmetic_standard_dev']):
            df['cv'] = df['arithmetic_standard_dev'] / (df['arithmetic_mean'] + 1e-8)
        
        # 5. Site metadata features
        if 'cbsa_name' in df.columns:
            df['is_urban'] = df['cbsa_name'].notna().astype(int)
        else:
            df['is_urban'] = 0
            
        if 'certification_indicator' in df.columns:
            df['has_certification'] = df['certification_indicator'].astype(str).apply(
                lambda x: 1 if 'Certified' in x else 0
            )
        else:
            df['has_certification'] = 0
        
        # 6. Extract features from available columns
        # Look for percentiles
        percentile_cols = [col for col in df.columns if 'percentile' in col.lower()]
        for col in percentile_cols[:5]:  # Use first 5 percentile columns
            if col in df.columns:
                df[f'{col}_normalized'] = df[col] / (df[col].max() + 1e-8)
        
        # Look for max values
        max_cols = [col for col in df.columns if 'max' in col.lower() and 'datetime' not in col.lower()]
        for col in max_cols[:3]:  # Use first 3 max columns
            if col in df.columns:
                df[f'{col}_ratio'] = df[col] / (df['arithmetic_mean'] + 1e-8)
        
        print(f"Feature engineering complete. Total columns: {len(df.columns)}")
        return df
    
    def create_sequences(self, df):
        """Create time sequences for each location-pollutant combination"""
        print("\nCreating sequences...")
        
        sequences = []
        sequence_info = []
        
        # Sort by location and time if available
        sort_cols = []
        if 'location_id' in df.columns:
            sort_cols.append('location_id')
        if 'year' in df.columns:
            sort_cols.append('year')
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        # Select numeric columns for sequences (exclude IDs and text)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns
        exclude_cols = ['state_code', 'county_code', 'site_num', 'year', 
                       'observation_count', 'observation_percent', 'pollutant_code']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Take top 15 features to keep manageable
        if len(numeric_cols) > 15:
            # Select features with highest variance
            variances = df[numeric_cols].var()
            numeric_cols = variances.nlargest(15).index.tolist()
        
        print(f"Selected {len(numeric_cols)} numeric features")
        
        if len(numeric_cols) < 5:
            print("ERROR: Not enough numeric features!")
            return np.array([]), []
        
        # Group by location if available, otherwise use entire dataset
        if 'location_id' in df.columns:
            groups = df.groupby('location_id')
        else:
            # Use entire dataframe as one group
            groups = [('all', df)]
        
        total_sequences = 0
        
        for group_name, group in groups:
            # Extract the sequence data
            data = group[numeric_cols].values.astype(np.float32)
            
            # Create sliding windows
            if len(data) >= self.seq_len:
                for i in range(0, len(data) - self.seq_len + 1, self.stride):
                    seq = data[i:i + self.seq_len]
                    sequences.append(seq)
                    sequence_info.append({
                        'location': str(group_name),
                        'start_idx': i,
                        'features': numeric_cols
                    })
                    total_sequences += 1
        
        if total_sequences == 0:
            print("WARNING: No sequences created! Trying alternative approach...")
            # Try creating sequences from entire dataset
            data = df[numeric_cols].values.astype(np.float32)
            
            if len(data) >= self.seq_len:
                for i in range(0, len(data) - self.seq_len + 1, self.stride * 2):
                    seq = data[i:i + self.seq_len]
                    sequences.append(seq)
                    sequence_info.append({
                        'location': 'all',
                        'start_idx': i,
                        'features': numeric_cols
                    })
                    total_sequences += 1
        
        if total_sequences == 0:
            print("WARNING: Still no sequences! Reducing sequence length...")
            # Reduce sequence length
            self.seq_len = min(self.seq_len, len(data) // 2)
            if self.seq_len >= 5:
                for i in range(0, len(data) - self.seq_len + 1, self.stride * 3):
                    seq = data[i:i + self.seq_len]
                    sequences.append(seq)
                    sequence_info.append({
                        'location': 'all',
                        'start_idx': i,
                        'features': numeric_cols
                    })
                    total_sequences += 1
        
        if total_sequences > 0:
            sequences = np.array(sequences)
            print(f"SUCCESS: Created {len(sequences)} sequences of shape {sequences.shape}")
        else:
            sequences = np.array([])
            print("ERROR: Failed to create sequences!")
        
        return sequences, sequence_info
    
    def process_features(self, sequences, sequence_info):
        """Process and scale features appropriately"""
        print("\nProcessing features...")
        
        if len(sequences) == 0:
            return np.array([]), {}
        
        # Identify feature groups for different scaling strategies
        all_features = sequence_info[0]['features'] if sequence_info else []
        
        # Group features by type
        self.feature_groups = {
            'pollution_levels': [i for i, f in enumerate(all_features) 
                               if any(x in f.lower() for x in ['mean', 'value', 'percentile', 'max', 'ratio'])],
            'statistics': [i for i, f in enumerate(all_features) 
                          if any(x in f.lower() for x in ['std', 'dev', 'cv', 'normalized'])],
            'geographic': [i for i, f in enumerate(all_features) 
                          if any(x in f.lower() for x in ['lat', 'lon'])],
            'metadata': [i for i, f in enumerate(all_features) 
                        if any(x in f.lower() for x in ['urban', 'cert', 'year'])]
        }
        
        # Ensure all features are in at least one group
        all_indices = set(range(len(all_features)))
        grouped_indices = set()
        for indices in self.feature_groups.values():
            grouped_indices.update(indices)
        
        ungrouped = all_indices - grouped_indices
        if ungrouped:
            self.feature_groups['other'] = list(ungrouped)
        
        # Apply different scaling strategies
        scaled_sequences = np.zeros_like(sequences)
        scalers = {}
        
        for group_name, indices in self.feature_groups.items():
            if indices:  # If group has features
                group_data = sequences[:, :, indices].reshape(-1, len(indices))
                
                # Choose scaler based on group
                if group_name in ['pollution_levels', 'statistics']:
                    # Robust scaling for pollution levels (outlier resistant)
                    scaler = RobustScaler(quantile_range=(5, 95))
                else:
                    # Standard scaling for others
                    scaler = StandardScaler()
                
                # Fit and transform
                try:
                    scaled_group = scaler.fit_transform(group_data)
                    scaled_sequences[:, :, indices] = scaled_group.reshape(
                        sequences.shape[0], sequences.shape[1], len(indices)
                    )
                    
                    scalers[group_name] = {
                        'scaler': scaler,
                        'feature_indices': indices,
                        'feature_names': [all_features[i] for i in indices]
                    }
                except Exception as e:
                    print(f"Warning: Failed to scale group {group_name}: {e}")
                    # Use original data
                    scaled_sequences[:, :, indices] = sequences[:, :, indices]
        
        # Clean any NaN/Inf values
        scaled_sequences = np.nan_to_num(scaled_sequences, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return scaled_sequences, scalers
    
    def save_data(self, sequences, sequence_info, scalers):
        """Save processed data"""
        print("\nSaving processed data...")
        
        if len(sequences) == 0:
            print("ERROR: No sequences to save!")
            return None, None, None
        
        # Split into train/val/test
        n_sequences = len(sequences)
        indices = np.random.permutation(n_sequences)
        
        train_idx = indices[:int(0.7 * n_sequences)]
        val_idx = indices[int(0.7 * n_sequences):int(0.85 * n_sequences)]
        test_idx = indices[int(0.85 * n_sequences):]
        
        train_data = sequences[train_idx]
        val_data = sequences[val_idx]
        test_data = sequences[test_idx]
        
        # Save numpy arrays
        np.save(os.path.join(self.output_dir, 'train.npy'), train_data)
        np.save(os.path.join(self.output_dir, 'val.npy'), val_data)
        np.save(os.path.join(self.output_dir, 'test.npy'), test_data)
        
        # Save metadata
        metadata = {
            'n_sequences': n_sequences,
            'sequence_length': self.seq_len,
            'n_features': sequences.shape[2],
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'feature_groups': {k: len(v) for k, v in self.feature_groups.items()}
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(os.path.join(self.output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(scalers, f)
        
        # Save sequence info (only first 1000 to avoid large files)
        if sequence_info:
            with open(os.path.join(self.output_dir, 'sequence_info.pkl'), 'wb') as f:
                pickle.dump(sequence_info[:1000], f)
        
        # Save feature names
        if sequence_info:
            with open(os.path.join(self.output_dir, 'feature_names.txt'), 'w') as f:
                for feature in sequence_info[0]['features']:
                    f.write(f"{feature}\n")
        
        print(f"SUCCESS: Data saved to {self.output_dir}")
        print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        return train_data, val_data, test_data
    
    
    def run(self):
        """Run complete preprocessing pipeline"""
        print("=" * 50)
        print("AIR QUALITY DATA PREPROCESSING")
        print("=" * 50)
        
        # 1. Load and clean
        df = self.load_and_clean()
        
        if len(df) == 0:
            print("ERROR: No data after cleaning!")
            return None
        
        # 2. Engineer features
        df = self.engineer_features(df)
        
        # 3. Create sequences
        sequences, sequence_info = self.create_sequences(df)
        
        if len(sequences) == 0:
            print("ERROR: No valid sequences created!")
            return None
        
        # 4. Process and scale features
        scaled_sequences, scalers = self.process_features(sequences, sequence_info)
        
        # 5. Save data
        train, val, test = self.save_data(scaled_sequences, sequence_info, scalers)
        
        print("\nPREPROCESSING COMPLETE!")
        print(f"Total sequences: {len(sequences)}")
        print(f"Sequence shape: {sequences.shape}")
        
        return train, val, test



def main():
    parser = argparse.ArgumentParser(description='Preprocess air quality data for TimeGAN')
    parser.add_argument('--input', type=str, required=True, help='Path to air1.csv')
    parser.add_argument('--output', type=str, default='data/processed/air_quality', 
                       help='Output directory')
    parser.add_argument('--seq_len', type=int, default=24, help='Sequence length')
    parser.add_argument('--stride', type=int, default=6, help='Stride for sliding window')
    
    args = parser.parse_args()
    
    preprocessor = AirQualityPreprocessor(
        input_csv=args.input,
        output_dir=args.output,
        seq_len=args.seq_len,
        stride=args.stride
    )
    
    result = preprocessor.run()
    
    if result is None:
        print("\nPREPROCESSING FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()