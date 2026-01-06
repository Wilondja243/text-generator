import torch
from torch import nn
import torchvision.transforms as transforms


class TextGeneration(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            output_dim,
            pad_idx):
        
        """
            Initialize the text generation model with embedding, LSTM, and output layers.

            Args:
                vocab_size (int): Size of the vocabulary
                embedding_dim (int): Dimension of the embedding vectors
                hidden_dim (int): Number of hidden units in the LSTM
                num_layers (int): Number of LSTM layers
                output_dim (int): Dimension of the output (usually vocab size)
                pad_idx (int): Index used for padding tokens
        """

        super().__init__()

        # Embedding layer: converts token IDs into dense vectors
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        # LSTM layer: processes sequences of embeddings
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=False
        )

        # Fully connected layer: maps LSTM outputs to vocab probabilities
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, text):
        
        # Convert token IDs to embeddings
        embedding = self.embedding(text)

        # Pass embeddings through LSTM
        output, (hidden, cell) = self.rnn(embedding)

        # Map LSTM outputs to the output dimension (vocabulary size)
        prediction = self.fc(output)

        return prediction
