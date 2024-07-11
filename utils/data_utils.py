import numpy as np
import os
import re

def load_data(file_path):
    """Load text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_text(text):
    """Preprocess text by removing digits, special characters, converting to lowercase, and trimming whitespaces."""
    text = re.sub(r'\d+', '', text)  # Remove all digits
    text = re.sub(r'[.,]', '', text)  # Remove specific punctuation
    text = re.sub(r'[^\w\s]', '', text)  # Remove all special characters except whitespaces
    text = text.lower()  # Convert text to lowercase
    text = '\n'.join([line.strip() for line in text.split('\n')])  # Remove leading and trailing whitespace from each line
    chars = sorted(set(text))  # Get unique characters in the text
    char_to_idx = {ch: i for i, ch in enumerate(chars)}  # Create a mapping from characters to indices
    idx_to_char = {i: ch for i, ch in enumerate(chars)}  # Create a mapping from indices to characters
    return text, char_to_idx, idx_to_char

def create_sequences(text, char_to_idx, seq_length):
    """Create input sequences and corresponding labels for training."""
    X = []  # List to store input sequences
    y = []  # List to store corresponding labels
    for i in range(0, len(text) - seq_length):
        try:
            X.append([char_to_idx[char] for char in text[i:i + seq_length]])  # Convert characters to indices
            y.append(char_to_idx[text[i + seq_length]])  # Get the label for the sequence
        except KeyError as e:
            print(f"KeyError: {e} at position {i}. Skipping this sequence.")  # Debug statement
    return np.array(X), np.array(y)  # Return input sequences and labels as numpy arrays

# Example usage
# file_path = '/content/poetry_generator/data/truyen_kieu.txt'  # Adjust the path as needed
# text_ = load_data(file_path)
# print(f"Loaded text: {text_[:100]}...")  # Print first 100 characters to verify

# text_, char_to_idx, idx_to_char = preprocess_text(text_)
# print(f"char_to_idx: {char_to_idx}")
# print(f"idx_to_char: {idx_to_char}")

# seq_length = 100  # Example sequence length
# X, y = create_sequences(text_, char_to_idx, seq_length)
# print(f"Number of sequences: {len(X)}")
# print(f"First sequence: {X[0] if len(X) > 0 else 'No sequences generated'}")
# print(f"First label: {y[0] if len(y) > 0 else 'No labels generated'}")
