import torch

def evaluate_model(model, dataloader, loss_fn, device, vocab_size):
    # 2. Set the model to evaluation mode
    model.eval()

    # 3. Initialize a variable to accumulate the loss
    total_eval_loss = 0

    # 4. Disable gradient calculations during evaluation
    with torch.no_grad():
        # 5. Iterate through each batch in the dataloader
        for batch in dataloader:
            # 6. Move input_ids and attention_mask to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 7. Perform a forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 8. Calculate the loss for next token prediction
            # Shift targets for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Flatten the tensors for CrossEntropyLoss
            loss = loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            # 9. Add the loss to total_eval_loss
            total_eval_loss += loss.item()

    # 10. Calculate the average evaluation loss
    avg_eval_loss = total_eval_loss / len(dataloader)

    # 11. Print or return the avg_eval_loss
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")

    # 12. (Optional) Set the model back to training mode if further training is expected
    model.train()

    return avg_eval_loss
