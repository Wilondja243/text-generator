import os
import torch
from torch import nn
from torch.functional import F
import torchvision.transforms as transforms
from src.nn import TextGeneration
from src.tokenizer import CustomData
from src.config_load import config_load


data_handler = CustomData()
dataloader = data_handler.get_dataloader()
tokenizer = data_handler.tokenizer
params = config_load()


VOCAB_SIZE = tokenizer.vocab_size
PAD_IDX = tokenizer.pad_token_id

EMBEDDING_DIM = params["model"]["embedding_dim"]
HIDDEN_DIM = params["model"]["hidden_dim"]
NUM_LAYERS = params["model"]["num_layers"]
OUTPUT_DIM = VOCAB_SIZE

EPOCH = params["training"]["epoch"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TextGeneration(
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    OUTPUT_DIM,
    PAD_IDX
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=params["training"]["learning_rate"])

# Training loop:
# Iterates over the dataset in batches, performs forward and backward passes,
# updates model weights using the optimizer
def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = params["training"]["epoch_loss"]

    for batch in dataloader:
        src = batch['input_ids'].to(device)
        trg = batch['labels'].to(device)

        optimizer.zero_grad()

        src = src.permute(1, 0)
        trg = trg.permute(1, 0)

        output = model(src)

        output_sliced = output[:-1].contiguous()
        
        trg_sliced = trg[1:].contiguous()
        
        min_len = min(output_sliced.shape[0], trg_sliced.shape[0])
        output_sliced = output_sliced[:min_len]
        trg_sliced = trg_sliced[:min_len]
        
        output_dim = output_sliced.shape[-1]
        output_flat = output_sliced.view(-1, output_dim) 
        trg_flat = trg_sliced.view(-1)

        loss = criterion(output_flat, trg_flat)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Loop that trains the model over the dataset for multiple epochs
for epoch in range(EPOCH):
    train_loss = train_loop(model, dataloader, optimizer, criterion, DEVICE)

    print(f"Epoch: {epoch+1:02} | Perte d'entrainement: {train_loss:.3f}")

folder = 'models'
os.makedirs(folder, exist_ok=True)

torch.save(model.state_dict, f"{folder}/gen_model.pth")

print("Model saved successfully")
