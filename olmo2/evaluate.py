import torch

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)) # Calculate perplexity from average loss

    return avg_loss, perplexity