from torch.utils.data import Dataset
import torch

class CodeDataset(Dataset):
    def __init__(self, data, vocab, max_sequence_length, bos_token_id, eos_token_id, pad_token_id, unk_token_id):
        self.data = data
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens']
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        input_ids = [self.bos_token_id] + token_ids
        target_ids = token_ids + [self.eos_token_id]

        def pad_or_truncate(ids, max_len):
            if len(ids) > max_len:
                return ids[:max_len]
            else:
                return ids + [self.pad_token_id] * (max_len - len(ids))

        effective_max_len = self.max_sequence_length - 1
        input_ids = pad_or_truncate(input_ids, effective_max_len)
        target_ids = pad_or_truncate(target_ids, effective_max_len)

        attention_mask = [1] * len(input_ids)
        try:
            first_pad_idx = input_ids.index(self.pad_token_id)
            attention_mask[first_pad_idx:] = [0] * (len(input_ids) - first_pad_idx)
        except ValueError: pass

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool)
        }