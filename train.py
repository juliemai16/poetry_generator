import argparse
import os
from utils.data_utils import load_data, preprocess_text, create_sequences
from model.rnn_model import build_model

def main(args):
    """Main function to load data, preprocess, and train the model."""
    print("Step 1: Loading data...")
    text = load_data(args.data_path)  # Load text data
    print("Data loaded successfully.")
    
    print("Step 2: Preprocessing text...")
    text, char_to_idx, idx_to_char = preprocess_text(text)  # Preprocess text and create mappings
    print(f"Text preprocessing completed. Number of unique characters: {len(char_to_idx)}")
    
    print(f"Step 3: Creating sequences with sequence length {args.seq_length}...")
    X, y = create_sequences(text, char_to_idx, args.seq_length)  # Create input sequences and labels
    print(f"Sequences created. Number of sequences: {len(X)}")
    
    print("Step 4: Building the model...")
    model = build_model(len(char_to_idx), args.seq_length, args.embedding_dim, args.lstm_units)  # Build the model
    print("Model built successfully.")
    
    print(f"Step 5: Training the model for {args.epochs} epochs...")
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size)  # Train the model
    print("Model training completed.")
    
    print(f"Step 6: Saving the model to {args.model_path}...")
    model.save(args.model_path)  # Save the trained model
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/truyen_kieu.txt', help='Path to the text data')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for training')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_path', type=str, default='model/poetry_model.h5', help='Path to save the trained model')
    
    args = parser.parse_args()
    main(args)  # Run the main function with parsed arguments
