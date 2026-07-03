
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch data to the specified device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()  # Zero out gradients before each batch

        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)

        # Calculate loss
        # For CrossEntropyLoss, logits should be (N, C) and labels (N)
        # N is total elements, C is number of classes (vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss