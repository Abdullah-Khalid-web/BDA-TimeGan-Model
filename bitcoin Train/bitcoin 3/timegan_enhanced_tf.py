# timegan_enhanced_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class EnhancedEmbedder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM layers
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.Bidirectional(
                layers.LSTM(hidden_dim // 2, return_sequences=True, 
                           kernel_regularizer=regularizers.l2(1e-5)),
                name=f"bilstm_{i}"
            )
            self.lstms.append(lstm)
        
        # Attention mechanism
        self.attention = layers.Attention(use_scale=True)
        self.dropout = layers.Dropout(dropout)
        self.dense = layers.Dense(hidden_dim, activation='tanh')
        
    def call(self, x, training=False):
        for lstm in self.lstms:
            x = lstm(x)
            x = self.dropout(x, training=training)
        
        # Self-attention
        attention_output = self.attention([x, x])
        x = tf.concat([x, attention_output], axis=-1)
        x = self.dense(x)
        return x

class EnhancedGenerator(tf.keras.Model):
    def __init__(self, z_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.dense_in = layers.Dense(hidden_dim, activation='relu')
        
        # Temporal convolutional network for better pattern generation
        self.conv_layers = []
        for i in range(2):
            conv = layers.Conv1D(
                hidden_dim, kernel_size=3, padding='same',
                activation='relu', kernel_regularizer=regularizers.l2(1e-5)
            )
            self.conv_layers.append(conv)
        
        # LSTM layers
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_dim, return_sequences=True,
                              kernel_regularizer=regularizers.l2(1e-5))
            self.lstms.append(lstm)
        
        # Skip connections
        self.dense_out = layers.Dense(hidden_dim, activation='tanh')
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, z, training=False):
        x = self.dense_in(z)
        
        # Convolutional processing
        for conv in self.conv_layers:
            x_residual = x
            x = conv(x)
            x = x + x_residual  # Skip connection
        
        # LSTM processing
        for lstm in self.lstms:
            x = lstm(x)
        
        x = self.dense_out(x)
        x = self.layer_norm(x)
        return x

class EnhancedDiscriminator(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.conv_layers = []
        
        # 1D Convolutional layers for local pattern detection
        for filters in [64, 128, 256]:
            conv = layers.Conv1D(
                filters, kernel_size=3, strides=2,
                padding='same', activation='leaky_relu'
            )
            self.conv_layers.append(conv)
        
        # Self-attention for global context
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        
        # LSTM for temporal dependencies
        self.lstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )
        
        # Output layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='leaky_relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(128, activation='leaky_relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
        
    def call(self, x, training=False):
        # Convolutional feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        # Attention mechanism
        attention_out = self.attention(x, x)
        x = x + attention_out  # Residual connection
        
        # LSTM processing
        x = self.lstm(x)
        
        # Classification head
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)

# Keep Supervisor and Recovery similar but add enhancements
class EnhancedSupervisor(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_dim, return_sequences=True,
                              kernel_regularizer=regularizers.l2(1e-5))
            self.lstms.append(lstm)
        self.dense = layers.Dense(hidden_dim, activation='tanh')
        
    def call(self, x, training=False):
        for lstm in self.lstms:
            x = lstm(x)
        return self.dense(x)

class EnhancedRecovery(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstms = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_dim, return_sequences=True)
            self.lstms.append(lstm)
        self.dense = layers.Dense(output_dim, activation='linear')
        
    def call(self, x, training=False):
        for lstm in self.lstms:
            x = lstm(x)
        return self.dense(x)