import argparse
import numpy as np
import tensorflow as tf
from utils.data_utils import preprocess_text, load_data

def sample(preds, temperature=1.0):
    """Sample an index from the probability array with temperature scaling."""
    preds = np.asarray(preds).astype('float64')  # Convert to float64 for precision
    preds = np.log(preds) / temperature  # Apply temperature scaling
    exp_preds = np.exp(preds)  # Exponentiate the predictions
    preds = exp_preds / np.sum(exp_preds)  # Normalize the predictions to get probabilities
    probas = np.random.multinomial(1, preds, 1)  # Sample an index based on probabilities
    return np.argmax(probas)  # Return the sampled index

def generate_text(model, seed_text, char_to_idx, idx_to_char, seq_length, length, temperature):
    """Generate text using the trained model."""
    seed_text = seed_text.lower()  # Convert seed text to lowercase
    generated_text = seed_text
    input_seq = [char_to_idx[char] for char in seed_text]

    # Pad or truncate the input sequence to match the required sequence length
    if len(input_seq) < seq_length:
        input_seq = [0] * (seq_length - len(input_seq)) + input_seq
    else:
        input_seq = input_seq[-seq_length:]
    
    for _ in range(length):
        input_seq_array = np.array(input_seq).reshape(1, -1)  # Prepare input sequence
        preds = model.predict(input_seq_array, verbose=0)[0]  # Get model predictions
        next_char = idx_to_char[sample(preds, temperature)]  # Sample the next character
        
        generated_text += next_char  # Append the character to the generated text
        input_seq.append(char_to_idx[next_char])  # Add the next character to the input sequence
        input_seq = input_seq[1:]  # Keep the sequence length consistent
    
    return generated_text  # Return the generated text

def main(args):
    """Main function to load model and generate text."""
    print("Step 1: Loading data...")
    text = load_data(args.data_path)  # Load text data
    print("Data loaded successfully.")
    
    print("Step 2: Preprocessing text...")
    text, char_to_idx, idx_to_char = preprocess_text(text)  # Preprocess text and create mappings
    print("Text preprocessing completed.")
    
    print(f"Step 3: Loading the model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)  # Load the trained model
    print("Model loaded successfully.")
    
    print(f"Step 4: Generating text with seed '{args.seed_text}'...")
    generated_text = generate_text(model, args.seed_text, char_to_idx, idx_to_char, args.seq_length, args.length, args.temperature)  # Generate text
    
    print("Generated text:")
    print(generated_text)  # Print the generated text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/truyen_kieu.txt', help='Path to the text data')
    parser.add_argument('--model_path', type=str, default='model/poetry_model.h5', help='Path to the trained model')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for generating text')
    parser.add_argument('--seed_text', type=str, default='Trăm năm trong cõi người ta', help='Seed text for generating poetry')
    parser.add_argument('--length', type=int, default=100, help='Length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    
    args = parser.parse_args()
    main(args)  # Run the main function with parsed arguments
