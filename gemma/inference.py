import torch

def generate_text(model, tokenizer, prompt, max_new_tokens, device, eos_token_id=None, model_max_seq_len=None):
    # 2. Set the model to evaluation mode
    model.eval()

    # 3. Tokenize the prompt and move to device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Initialize attention mask for the initial prompt
    attention_mask = torch.ones_like(input_ids).to(device)

    # Store the generated tokens (initially just the prompt tokens)
    generated_ids = input_ids

    # 5. Start a loop for text generation
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 5a. Pass the current input_ids and attention_mask through the model
            # If the current sequence length exceeds model_max_seq_len, truncate it
            current_seq_len = generated_ids.size(1)
            if model_max_seq_len and current_seq_len >= model_max_seq_len:
                # Only consider the last part of the sequence that fits the model's max_seq_len
                # This is a common strategy for very long sequence generation, though simpler models might just stop.
                # For this implementation, we will stop if max_seq_len is hit, similar to truncation logic.
                print(f"Warning: Sequence length reached model_max_seq_len ({model_max_seq_len}). Stopping generation.")
                break

            # Generate logits
            logits = model(input_ids=generated_ids, attention_mask=attention_mask)

            # 5b. Get the logits for the *last* token in the sequence
            next_token_logits = logits[:, -1, :]

            # 5c. Apply torch.argmax to these last token logits to get the next_token_id (greedy decoding)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # unsqueeze to make it (1,1)

            # 5d. Append next_token_id to generated_ids
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Extend attention_mask for the new token
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)

            # 5e. If the next_token_id is the eos_token_id, break the loop.
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    # 6. Decode the generated_ids back into text
    # Exclude the initial prompt tokens from the output if desired. Here we decode the whole sequence.
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text
