# Text Generator

Text Generator is an **educational project** built from scratch using **PyTorch**.  
It demonstrates the complete **text generation pipeline**, from **data cleaning and tokenization** to **model training** and **text prediction**.  
The goal of this project is to **deeply understand how text generation models work internally**.

---

## Features

- Load and clean text data from CSV
- Tokenization using `transformers` tokenizer
- Custom PyTorch LSTM model for text generation
- Training loop with configurable hyperparameters
- Interactive text prediction via console
- Saving and loading model weights
- Configurable through `config.yaml`

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/text-generator.git
cd text-generator

2. Create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate 
venv\Scripts\activate

```

3. Install dependencies:

```
pip install -r requirements.txt


## Usage

Interactive prediction:

```
python run.py
```

## Configuration

All hyperparameters are stored in configs/config.yaml:

- Model parameters: embedding_dim, hidden_dim, num_layers

- Training parameters: batch_size, learning_rate, epochs

- Others: pad_idx, vocab_size, etc.


Contributing

This is an educational project, feel free to experiment with:

- Different datasets

- Other tokenizers or models (GRU, Transformer)

- Training strategies and hyperparameters


## Licence

This project is released under the MIT License.

---

Si tu veux, je peux aussi te faire **une version plus courte et “GitHub-ready”**, qui tient **en une page et reste attractive** pour quelqu’un qui découvre ton repo.  
