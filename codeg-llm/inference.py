import torch
from utils import generate_causal_mask,tokens_to_code
import torch.nn.functional as F

def generate_code_sequence(model, vocab, id_to_token, max_sequence_length, device, bos_token_id, eos_token_id, pad_token_id, unk_token_id, temperature=1.0, top_k=None, top_p=None, start_prompt_tokens=None, copy_mechanism_enabled=False, copy_score_boost=0.5):
    """
    Generates a code sequence using the trained Transformer Decoder model.
    Incorporates greedy decoding, top-k, top-p sampling strategies, and an optional copy mechanism.

    Args:
        model: The trained TransformerDecoderModel.
        vocab: Token to ID mapping.
        id_to_token: ID to Token mapping.
        max_sequence_length: Maximum length of the generated sequence.
        device: The device (cpu/cuda) where the model is loaded.
        bos_token_id: ID of the <BOS> token.
        eos_token_id: ID of the <EOS> token.
        pad_token_id: ID of the <PAD> token.
        unk_token_id: ID of the <UNK> token.
        temperature: Softmax temperature for sampling (1.0 for greedy).
        top_k: If not None and > 0, sample from the top k most likely tokens.
        top_p: If not None and between 0 and 1, sample from the smallest set of tokens whose cumulative probability exceeds p.
        start_prompt_tokens: Optional list of token strings to start generation from (excluding BOS).
        copy_mechanism_enabled (bool): If True, boost logits of tokens present in the initial prompt.
        copy_score_boost (float): The amount to boost logits for copied tokens if copy_mechanism_enabled is True.

    Returns:
        A list of generated token IDs (excluding BOS/EOS) and the human-readable code string.
    """
    model.eval()
    generated_ids = []

    # Prepare initial input. If a start_prompt_tokens is provided, tokenize it.
    input_ids = [bos_token_id]
    if start_prompt_tokens:
        # Store prompt tokens for copy mechanism, converting to lowercase as per normalization
        # Filter out special tokens from prompt tokens to avoid boosting them unnecessarily
        copy_tokens_in_vocab = []
        for token_str in start_prompt_tokens:
            normalized_token = token_str.lower() # Apply same normalization as during training
            if normalized_token in vocab and normalized_token not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                copy_tokens_in_vocab.append(vocab[normalized_token])
            input_ids.append(vocab.get(normalized_token, unk_token_id))
        copy_tokens_in_vocab = list(set(copy_tokens_in_vocab)) # Unique token IDs from prompt

    # Convert to tensor and move to device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_sequence_length - len(input_ids)): # Generate tokens up to max_sequence_length
            # Generate causal mask for the current input length
            current_seq_len = input_tensor.size(1)
            current_causal_mask = generate_causal_mask(current_seq_len).to(device)

            # Get model predictions
            output_logits = model(input_tensor, tgt_mask=current_causal_mask)

            # Get the logits for the last token in the sequence
            next_token_logits = output_logits[:, -1, :] / temperature

            # --- Copy Mechanism Implementation ---
            if copy_mechanism_enabled and start_prompt_tokens:
                for token_id_to_boost in copy_tokens_in_vocab:
                    if token_id_to_boost < next_token_logits.size(-1): # Ensure ID is within vocab size
                        next_token_logits[:, token_id_to_boost] += copy_score_boost

            # Apply sampling strategy
            if top_k is not None and top_k > 0:
                # Top-K sampling
                values, _ = torch.topk(next_token_logits, top_k)
                min_value = values[:, -1].unsqueeze(1) # Smallest value in top-k
                next_token_logits[next_token_logits < min_value] = float('-inf')

            if top_p is not None and 0 < top_p < 1:
                # Top-P (nucleus) sampling. Note: top_k takes precedence if both are set.
                # Sort all probabilities from largest to smallest
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold (top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')

            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample next token
            predicted_token_id = torch.multinomial(probs, num_samples=1).item()

            if predicted_token_id == eos_token_id or predicted_token_id == pad_token_id:
                break # Stop if EOS or PAD token is predicted

            generated_ids.append(predicted_token_id)

            # Append the predicted token to the input for the next step
            input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_token_id]], dtype=torch.long, device=device)], dim=1)

    # Convert generated IDs back to human-readable code
    special_token_ids_set = {pad_token_id, bos_token_id, eos_token_id, unk_token_id}
    human_readable_code = tokens_to_code(generated_ids, id_to_token, special_token_ids_set)

    return generated_ids, human_readable_code