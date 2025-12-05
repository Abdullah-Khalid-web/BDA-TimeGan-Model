# BDA-TimeGan-Model

Stock Prices & Air Quality TimeGAN Pipeline
A comprehensive pipeline for generating synthetic time-series data using TimeGAN (Time-series Generative Adversarial Networks). This project supports multiple domains including Stock Prices and Air Quality data.

📋 Overview
This project provides a complete pipeline for:

Preprocessing raw time-series data (stock prices, air quality, etc.)

Training TimeGAN models to generate synthetic time-series data

Evaluating synthetic data quality

Analyzing results through multiple visualization techniques

Diagnostic analysis for model debugging and improvement
```
🏗️ Project Structure
For Stock Prices:
text
Stock_Prices_GAN_Training/
├── data/
│   ├── processed/stock_prices/          # Processed stock data
│   │   ├── features.txt                # Feature names
│   │   ├── meta.json                   # Dataset metadata
│   │   ├── scalers.pkl                 # Scaler objects
│   │   ├── test.npy                    # Test dataset
│   │   ├── train.npy                   # Training dataset
│   │   └── val.npy                     # Validation dataset
│   └── raw/                           # Raw input data
│
├── outputs/
│   ├── checkpoints/                   # Model checkpoints
│   ├── embeddings/                    # Feature embeddings
│   ├── eval/                          # Evaluation results
│   ├── figures/                       # Generated plots
│   └── synth/                         # Synthetic datasets
│
├── logs/                              # Training logs
│   └── wgan_gp_20251203-150433/      # Example log directory
│
└── src/                               # Source code
    ├── preprocess_stock_prices.py     # Stock data preprocessing
    ├── timegan_tf.py                  # TimeGAN implementation
    ├── train_timegan_adversarial_tf.py # Training script
    ├── evaluate_synth.py              # Evaluation script
    ├── generate_and_save.py           # Synthetic data generation
    └── various notebooks for analysis
```
```
For Air Quality:
text
Air Quality/air quality v1||v2
├── data/
│   ├── processed/air_quality/          # Processed air quality data
│   │   ├── train.npy                   # Training dataset
│   │   ├── val.npy                     # Validation dataset
│   │   └── test.npy                    # Test dataset
│   └── raw/                           # Raw input data
│
├── checkpoints/air_quality/           # Model checkpoints
├── outputs/synthetic_air_quality/     # Generated outputs
├── analysis_results/                  # Analysis outputs
│   ├── visualizations/                # Plots and charts
│   ├── comprehensive_report.txt       # Detailed analysis
│   └── analysis_results.json          # Structured results
│
└── source files for air quality pipeline
```

🚀 Quick Start
Installation
bash
# Clone the repository
git clone <repository-url>
cd BDA-TIMEGAN-MODEL

# Install required packages
pip install -r requirements.txt

# For TensorFlow with GPU support (optional)
pip install tensorflow-gpu
Run Complete Pipeline
For Stock Prices:
bash
# Navigate to stock prices directory
cd Stock_Prices_GAN_Training

# Run complete pipeline
python src/preprocess_stock_prices.py    # Step 1: Preprocess
python src/train_timegan_adversarial_tf.py  # Step 2: Train
python src/evaluate_synth.py             # Step 3: Evaluate
python src/generate_and_save.py          # Step 4: Generate
For Air Quality:
bash
# Navigate to main directory
cd BDA-TIMEGAN-MODEL

# Run complete pipeline
python air_quality_analysis.py           # Complete pipeline
python train_timegan_improved.py         # Training only
python evaluate_air_quality.py           # Evaluation only
🔧 Individual Components
1. Data Preprocessing
Stock Prices:
bash
python src/preprocess_stock_prices.py
Processes raw stock price data

Handles missing values, normalization

Creates train/val/test splits

Saves processed data in data/processed/stock_prices/

Air Quality:
bash
python preprocess_air_quality.py
Processes air quality sensor data

Handles temporal alignment

Feature engineering for pollutants

Saves processed data in data/processed/air_quality/

2. TimeGAN Training
Stock Prices:
bash
python src/train_timegan_adversarial_tf.py
Advanced TimeGAN with WGAN-GP loss

Multiple LSTM layers with attention

Spectral normalization

Logging to TensorBoard

Air Quality:
bash
python train_timegan_improved.py
Enhanced reconstruction loss (λ_rec = 10.0)

Embedding loss for temporal relationships

Autocorrelation loss for pattern preservation

Gradient penalty for stability

3. Evaluation
Stock Prices:
bash
python src/evaluate_synth.py
Statistical similarity metrics

Distribution comparisons (KS test)

Temporal pattern analysis

Feature space metrics (PCA, t-SNE)

Air Quality:
bash
python evaluate_air_quality.py
Mean correlation and MAE

KS test p-values

Auto-correlation similarity

Silhouette scores

4. Analysis & Visualization
Available Notebooks (Stock Prices):
bash
# Run Jupyter notebooks for analysis
jupyter notebook src/analysis_real_vs_synth.ipynb        # Real vs Synthetic comparison
jupyter notebook src/compute_embeddings.ipynb            # Embedding computation
jupyter notebook src/plot_embeddings_tsne_umap.ipynb     # Dimensionality reduction
jupyter notebook src/advanced_diagnostics_tf.ipynb       # Model diagnostics
jupyter notebook src/plot_side_by_side_panels.ipynb      # Visualization panels
Available Scripts (Air Quality):
bash
# Generate comprehensive analysis
python analyze_results.py                # Complete analysis
python generate_visualizations.py        # Create all plots
⚙️ Configuration
Stock Prices Configuration:
Key parameters in src/timegan_tf.py:

python
config = {
    'seq_len': 60,              # Sequence length
    'feature_dim': 10,          # Number of features (OHLCV + indicators)
    'hidden_dim': 128,          # Hidden layer dimension
    'z_dim': 20,                # Latent space dimension
    'batch_size': 64,           # Training batch size
    'epochs': 1000,             # Training epochs
    'lambda_gp': 10.0,          # Gradient penalty weight
    # ... more parameters
}
Air Quality Configuration:
Key parameters in train_timegan_improved.py:

python
config = {
    'seq_len': 24,              # 24-hour sequences
    'feature_dim': 15,          # 15 air quality features
    'hidden_dim': 256,          # Increased capacity
    'z_dim': 100,               # Larger latent space
    'lambda_rec': 10.0,         # CRITICAL: Reconstruction loss
    'lambda_emb': 100.0,        # Embedding loss
    'lambda_ac': 0.5,           # Autocorrelation loss
    'epochs': 500,              # Training epochs
    # ... more parameters
}
📊 Output Files
Stock Prices Outputs:
Synthetic Data: outputs/synth/ (CSV, NPY formats)

Model Checkpoints: outputs/checkpoints/

Evaluation Results: outputs/eval/

Visualizations: outputs/figures/

Embeddings: outputs/embeddings/

Air Quality Outputs:
Synthetic Data: outputs/synthetic_air_quality/

Model Checkpoints: checkpoints/air_quality_improved/

Evaluation Results: outputs/evaluation_results/

Visualizations: analysis_results/visualizations/

🎯 Use Cases
For Stock Prices:
Algorithmic Trading: Generate synthetic market scenarios

Risk Management: Stress testing with synthetic market conditions

Model Development: Test trading strategies without risking real capital

Data Augmentation: Expand limited historical data

For Air Quality:
Environmental Monitoring: Simulate pollution scenarios

Health Impact Studies: Generate data for epidemiological research

Policy Testing: Simulate effects of emission regulations

Sensor Network Design: Test sensor placement with synthetic data

🔍 Monitoring & Diagnostics
Training Monitoring:
bash
# For Stock Prices (TensorBoard)
tensorboard --logdir logs/wgan_gp_20251203-150433

# For Air Quality
# Check training_history.png in checkpoints directory
Quality Indicators:
Stock Prices:
Good: Correlation > 0.8, KS p-value > 0.05

Fair: Correlation 0.6-0.8, KS p-value > 0.01

Poor: Correlation < 0.6, KS p-value < 0.01

Air Quality:
Good: Overall score > 0.7

Fair: Overall score 0.4-0.7

Poor: Overall score < 0.4

🛠️ Troubleshooting
Common Issues:
Poor Synthetic Data Quality

Increase reconstruction loss weight (lambda_rec)

Check data normalization

Increase training epochs

Adjust latent dimension (z_dim)

Training Instability

Reduce learning rates

Increase gradient clipping

Use smaller batch sizes

Enable spectral normalization

Memory Issues

Reduce batch size

Generate data in smaller batches

Use n_samples parameter to limit generated data size

Domain-Specific Tips:
Stock Prices:
Use WGAN-GP loss for better stability

Include technical indicators as features

Handle market hours/non-trading hours appropriately

Consider volatility clustering

Air Quality:
Account for seasonal patterns

Handle missing sensor data

Consider spatial correlations between stations

Account for weather influences

📈 Performance Tuning
For Better Results:
Increase Model Capacity:

Increase hidden_dim (128 → 256 → 512)

Increase z_dim (20 → 100 → 200)

Add more LSTM layers

Adjust Loss Weights:

Stock: Focus on lambda_gp for stability

Air Quality: Focus on lambda_rec for fidelity

Training Strategy:

Increase epochs (1000+ for stocks, 500+ for air quality)

Use learning rate scheduling

Implement early stopping

📚 Analysis Notebooks
The project includes comprehensive Jupyter notebooks for in-depth analysis:

For Stock Prices:
advanced_diagnostics_tf.ipynb: Model diagnostics and debugging

analysis_real_vs_synth.ipynb: Statistical comparison

compute_embeddings.ipynb: Feature embedding computation

plot_embeddings_tsne_umap.ipynb: Dimensionality reduction visualization

compute_metric_tables_ci.ipynb: Confidence interval calculation

plot_side_by_side_panels.ipynb: Multi-panel visualization

For Air Quality:
Analysis scripts generate comprehensive reports

Visualization scripts create comparison plots

Evaluation scripts output structured JSON results

🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Create a Pull Request