import torch
from src.tokenizer import CustomData
from src.config_load import config_load
from src.predict import predict
from src.writer import writer

from src.train import (
    TextGeneration,
    VOCAB_SIZE,
    OUTPUT_DIM,
    PAD_IDX
)

# Load hyperparameters from config.yaml
params = config_load()

EMBEDDING_DIM = params["model"]["embedding_dim"]
HIDDEN_DIM = params["model"]["hidden_dim"]
NUM_LAYERS = params["model"]["num_layers"]

# Prepare dataset and tokenizer
data_handler = CustomData()
tokenizer = data_handler.tokenizer

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the device
model = TextGeneration(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    OUTPUT_DIM,
    PAD_IDX
).to(device)

def predict_output():

    """
        Main loop to interact with the model via console.
        Type 'quit', 'exit', or 'q' to stop.
    """

    while True:
        try:
            text = input("what can i do for you (type 'quit' to exit): ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            # Generate response and write output
            response = predict(model, tokenizer, text, device)
            writer(response)

        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    
    predict_output()
