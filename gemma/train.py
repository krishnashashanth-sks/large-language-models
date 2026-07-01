import tqdm
import torch

def train(num_epochs,dataloader,model,optimizer,loss_fn,vocab_size,device):
    # Loop over epochs
    for epoch in range(num_epochs):
        total_loss = 0
        # 3. Iterate through each batch in the dataloader
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # 4. Move input_ids and attention_mask to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 5. Zero out the gradients of the optimizer
            optimizer.zero_grad()

            # 6. Perform a forward pass
            # The model outputs logits for each token position
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 7. Calculate the loss for language modeling
            # Shift targets for next token prediction:
            #   Input: [t1, t2, t3, t4]
            #   Targets: [t2, t3, t4, <pad>] (or rather, model predicts t2 given t1, t3 given t2, etc.)
            #   So, logits for prediction should correspond to input_ids shifted by one.
            #   logits: (batch_size, seq_len, vocab_size)
            #   targets: (batch_size, seq_len)

            # For cross-entropy loss, we want to predict the next token. If input is [t1, t2, t3],
            # we want to predict t2 at position 0, t3 at position 1. The model's output at
            # position i (logits[i]) should predict token i+1.
            # Therefore, we take logits from all but the last token position
            # and targets from all but the first token position.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Flatten the tensors for CrossEntropyLoss which expects (N, C) and (N)
            loss = loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            # 8. Perform a backward pass to compute gradients
            loss.backward()

            # 9. Update the model's weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    print("Training loop completed.")