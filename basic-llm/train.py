def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train() # Set the model to training mode
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Clear previous gradients

        # Forward pass
        logits = model(input_ids)

        # Calculate loss. Shift labels for next token prediction
        # For example, if input_ids is [A, B, C], labels should be [B, C, <pad>] for predicting next token.
        # CrossEntropyLoss expects (N, C) for logits and (N) for labels.
        # Logits shape: (batch_size, sequence_length, vocab_size)
        # Labels shape: (batch_size, sequence_length)

        # Flatten logits and labels for CrossEntropyLoss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
