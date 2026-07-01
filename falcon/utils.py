import torch

def create_sequences(encoded_data, block_size):
    input_sequences = []
    target_sequences = []
    # Ensure there's enough data to create at least one sequence of block_size + 1
    if len(encoded_data) < block_size + 1:
        print(f"Warning: Not enough encoded data ({len(encoded_data)} tokens) to create sequences of block size {block_size}.")
        return torch.tensor([]), torch.tensor([])

    for i in range(0, len(encoded_data) - block_size):
        # Input sequence: from current position up to block_size
        input_seq = encoded_data[i : i + block_size]
        # Target sequence: shifted by one, from next position up to block_size
        target_seq = encoded_data[i + 1 : i + block_size + 1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    return torch.tensor(input_sequences), torch.tensor(target_sequences)
