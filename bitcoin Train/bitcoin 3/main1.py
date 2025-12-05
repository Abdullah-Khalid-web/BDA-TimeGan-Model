# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from timegan import timegan
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# import numpy as np
# from tslearn.metrics import dtw

# data = pd.read_csv("data/bitcoin.csv")
# data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# data = data.dropna()

# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(data)

# seq_len = 24  # number of time steps

# X = []
# for i in range(len(data_scaled) - seq_len):
#     X.append(data_scaled[i:i+seq_len])
# X = np.array(X)


# Sequence_length = 24  # hours or days
# Hidden_units = 24
# Learning_rate = 0.001
# Batch_size = 128
# Epochs = 500


# # Run training (timegan returns the generated/synthetic sequences)
# generated_data = timegan(X, hidden_dim=Hidden_units, num_layers=3, iterations=1000)
# synthetic_data = generated_data

# plt.plot(X[0][:,0], label="Real")
# plt.plot(generated_data[0][:,0], label="Fake")
# plt.legend()
# plt.show()


# sns.kdeplot(X[:, :, 0].flatten(), label="Real")
# sns.kdeplot(generated_data[:, :, 0].flatten(), label="Fake")
# plt.legend()
# plt.show()



# tsne = TSNE(n_components=2)
# embed_real = tsne.fit_transform(X.reshape(-1, X.shape[2]))
# embed_fake = tsne.fit_transform(generated_data.reshape(-1, generated_data.shape[2]))

# plt.scatter(embed_real[:,0], embed_real[:,1], alpha=0.3, label='Real')
# plt.scatter(embed_fake[:,0], embed_fake[:,1], alpha=0.3, label='Fake')
# plt.legend()
# plt.show()


# dist = dtw(X[0,:,0], generated_data[0,:,0])
# print("DTW distance:", dist)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from timegan import timegan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tslearn.metrics import dtw

# === Load and preprocess data ===
data = pd.read_csv("data/bitcoin.csv")
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Reduce dataset size
data_scaled = data_scaled[:50000]

seq_len = 24
X = []
for i in range(len(data_scaled) - seq_len):
    X.append(data_scaled[i:i+seq_len])

X = np.array(X)
if len(X) > 10000:
    idx = np.random.choice(len(X), 10000, replace=False)
    X = X[idx]

print("✅ Final dataset shape:", X.shape)

# === TimeGAN Parameters ===
parameters = {
    'module': 'lstm',
    'hidden_dim': 24,
    'num_layer': 3,
    'iterations': 1000,
    'batch_size': 128
}

# === Train TimeGAN ===
generated_data = timegan(X, parameters)

# === Visualization ===
plt.figure(figsize=(8, 4))
plt.plot(X[0][:, 0], label="Real")
plt.plot(generated_data[0][:, 0], label="Fake")
plt.legend()
plt.show()

sns.kdeplot(X[:, :, 0].flatten(), label="Real")
sns.kdeplot(generated_data[:, :, 0].flatten(), label="Fake")
plt.legend()
plt.show()

# === t-SNE Comparison ===
tsne = TSNE(n_components=2, perplexity=30)
embed_real = tsne.fit_transform(X.reshape(-1, X.shape[2]))
embed_fake = tsne.fit_transform(generated_data.reshape(-1, generated_data.shape[2]))

plt.figure(figsize=(6,6))
plt.scatter(embed_real[:, 0], embed_real[:, 1], alpha=0.3, label='Real')
plt.scatter(embed_fake[:, 0], embed_fake[:, 1], alpha=0.3, label='Fake')
plt.legend()
plt.show()

# === DTW Metric ===
dist = dtw(X[0, :, 0], generated_data[0, :, 0])
print("DTW distance:", dist)
