import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Embedding

def build_model(vocab_size, seq_length, embedding_dim, lstm_units):
    """Build and compile a BiLSTM model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length),  # Embedding layer
        Bidirectional(LSTM(lstm_units, return_sequences=True)),  # Bidirectional LSTM layer
        Bidirectional(LSTM(lstm_units)),  # Bidirectional LSTM layer
        Dense(vocab_size, activation='softmax')  # Dense layer with softmax activation
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')  # Compile the model
    return model  # Return the compiled model
