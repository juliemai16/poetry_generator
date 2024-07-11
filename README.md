# Poetry Generation with RNN

This project focuses on generating poetry using a Recurrent Neural Network (RNN) with a Bidirectional Long Short-Term Memory (BiLSTM) layer. The model is trained on text data and generates new text based on a seed input.

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Acknowledgements](#acknowledgements)

## Introduction

This project aims to create a poetry generator using RNN. It uses a BiLSTM network to capture the sequential nature of the text data. The model is trained on a corpus of text and can generate new poetry based on a given seed text.

## Architecture Overview

```plaintext

|   Input Layer    |  -> Shape: (batch_size, sequence_length)

         |
         v

| Embedding Layer  |  -> Shape: (batch_size, sequence_length, embedding_dim)

         |
         v

|   BiLSTM Layer   |  -> Shape: (batch_size, sequence_length, 2 * lstm_units)

         |
         v

|  Dense Layer     |  -> Shape: (batch_size, sequence_length, num_unique_chars)

         |
         v

|  Softmax Layer   |  -> Shape: (batch_size, sequence_length, num_unique_chars)

```

## Setup and Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/juliemai16/poetry_generator.git
    cd poetry_generator
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Prepare your dataset:

    Place your text data file in the data/ directory.

## Usage
1. **Train the Model**:
    ```bash
    python train.py --data_path data/truyen_kieu.txt --seq_length 100 --embedding_dim 256 --lstm_units 256 --epochs 20 --batch_size 64 --model_path model/poetry_model.h5
    ```

2. **Generating Text**:
    ```bash
    python generate.py --data_path data/truyen_kieu.txt --model_path model/poetry_model.h5 --seq_length 100 --seed_text "Trăm năm trong cõi người ta" --length 100 --temperature 1.0
    ```

## Project Structure
```
poetry_generator/
├── data/
│   └── truyen_kieu.txt        # Text data file
├── model/
│   └──rnn_model.py            # RNN-based model 
├── utils/
│   ├── data_utils.py          # Data loading and preprocessing utilities
│   └── __init__.py
├── train.py                   # Training script
├── generate.py                # Text generation script
├── requirements.txt           # Dependencies
└── README.md                  # Project description and instructions
```

## Acknowledgements
This project is inspired by various tutorials and examples from the ProtonX community and [Make Poem with AI](https://tiensu.github.io/blog/84_make_poem_with_ai/)
