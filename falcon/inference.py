import torch
import torch.nn.functional as F

def generate_text(model_config,model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, device='cpu'):
    model.eval()  # Set model to evaluation mode
    encoded_prompt = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0) # Add batch dimension

    generated_tokens = encoded_prompt.copy()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # If the input_ids exceed the model's max_position_embeddings,
            # truncate to the last `max_position_embeddings` tokens
            if input_ids.shape[1] > model_config['max_position_embeddings']:
                input_ids = input_ids[:, -model_config['max_position_embeddings']:]

            logits = model(input_ids)
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Append to generated sequence and update input_ids for the next step
            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            # Stop if an 'end of sequence' token is generated (optional, for real tokenizers)
            # For character-level, we might not have a specific EOS, so we rely on max_new_tokens

    return tokenizer.decode(generated_tokens)