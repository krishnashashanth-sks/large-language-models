class CharTokenizer:
    def __init__(self, text_data):
        self.chars = sorted(list(set(text_data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return [self.char_to_idx[c] for c in s]

    def decode(self, l):
        return ''.join([self.idx_to_char[i] for i in l])

    def __len__(self):
        return self.vocab_size