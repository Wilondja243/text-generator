import torch


def predict(model, tokenizer, question, device, max_len=50):

    """
        Generate a response from the model for a given input question.

        Args:
            model (nn.Module): The trained PyTorch model
            tokenizer (Tokenizer): The tokenizer used for encoding/decoding
            question (str): Input question string
            device (torch.device): Device to run the model on (CPU/GPU)
            max_len (int): Maximum number of tokens to generate

        Returns:
            str: The generated response as a decoded string
    """

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Tokenize the input question and convert to tensor
    encoding = tokenizer(question, return_tensors='pt', truncation=True)
    src_ids = encoding['input_ids'].to(device)

    # Permute to match model input shape if needed
    src_ids = src_ids.permute(1, 0)

    # Start target sequence with CLS token
    trg_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        for i in range(max_len):

            # Forward pass through the model
            output = model(trg_ids)

            # Select the most probable token at the current step
            pred_token_id = output.argmax(dim=-1)[-1, 0].item()

            # Stop if SEP token is generated
            if pred_token_id == tokenizer.sep_token_id:
                break

            # Append the predicted token to the target sequence
            trg_ids = torch.cat([trg_ids, torch.tensor([[pred_token_id]]).to(device)], dim=0)

    # Convert generated token IDs to list and decode to text
    generated_ids = trg_ids.squeeze(1).tolist()
    decoded_text = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)

    return decoded_text
