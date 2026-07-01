import torch
def train(num_epochs,train_dataloader,model,optimizer,loss_fn,gradient_clipping_value,device):
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            logits = model(input_ids)

            # Reshape for CrossEntropyLoss
            # logits: (batch_size, sequence_length, vocab_size) -> (batch_size * sequence_length, vocab_size)
            # labels: (batch_size, sequence_length) -> (batch_size * sequence_length)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training complete.")