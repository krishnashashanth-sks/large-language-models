import torch

def evaluate(model, dataloader, loss_fn, device):
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # Disable gradient calculation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)

            # Flatten logits and labels for CrossEntropyLoss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)
