import torch
import torch.nn.functional as F

def generate(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=0,device=torch.device("cpu")):
    model.eval() # Set model to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # The custom model does not have a `generate` method like Hugging Face models.
    generated_ids = input_ids
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated_ids) # Get logits for the current sequence
        
        # Get logits for the last token
        next_token_logits = logits[:, -1, :]

        if temperature == 0.0 or top_k == 0: # Greedy decoding or no sampling
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        else: # Sampling
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            # Apply top_k filtering
            if top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(next_token_logits < min_value, torch.full_like(next_token_logits, float('-inf')), next_token_logits)
            
            # Sample from the distribution
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)
        
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text