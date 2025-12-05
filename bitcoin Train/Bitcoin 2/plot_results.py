import matplotlib.pyplot as plt
import json
import os

# Load evaluation summary
with open("outputs/eval/eval_report.json") as f:
    report = json.load(f)

metrics = {
    "MMD": report["mmd_rbf"],
    "DTW": report["avg_dtw"],
    "Volatility Similarity": report["volatility_similarity"],
    "Kurtosis Ratio": report["kurtosis_ratio"],
    "Predictive MSE (real)": report["mse_predict_real_trained"],
    "Predictive MSE (synth)": report["mse_predict_synth_trained"]
}

plt.figure(figsize=(10,6))
plt.bar(metrics.keys(), metrics.values())
plt.xticks(rotation=30)
plt.title("TimeGAN Evaluation Metrics")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/eval/metrics_bar_chart.png")
plt.show()
