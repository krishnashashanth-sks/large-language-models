import torch

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, device='cpu'):
    model.eval()  # Set the model to evaluation mode

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_ids = input_ids

    for _ in range(max_new_tokens):
        # Get logits for the next token
        with torch.no_grad():
            # We only need the logits for the last token to predict the next one
            # The model is designed to handle the full sequence and apply causal masking internally.
            outputs = model(generated_ids)
            logits = outputs[:, -1, :]  # Logits for the last token in the sequence

        # Apply temperature
        if temperature == 0.0:
            # Greedy sampling
            next_token_logits = logits
        else:
            # Apply temperature and sample
            next_token_logits = logits / temperature

        # Sample the next token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to the generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text