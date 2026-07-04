import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch
import random
from torch.utils.data import  DataLoader
from model import TransformerDecoderModel
from dataset import CodeDataset
from inference import generate_code_sequence
from utils import normalize_tokens

dummy_tokenized_data_raw = [
    {'tokens': ['def', 'calculate_sum', '(', 'a', ',', 'b', ')', ':', 'result', '=', 'a', '+', 'b', ';', 'return', 'result']}, {'tokens': ['x', '=', '100', ';', 'y', '=', '200', ';', 'print', '(', 'x', '+', 'y', ')']},
    {'tokens': ['if', '(', 'temperature', '>', '25', ')', ':', 'print', '(', '"Hot"', ')', 'else', ':', 'print', '(', '"Cold"', ')']},
    {'tokens': ['for', 'i', 'in', 'range', '(', '5', ')', ':', 'print', '(', 'i', ')']},
    {'tokens': ['count', '=', '0', ';', 'while', '(', 'count', '<', '3', ')', ':', 'print', '(', '"Counting"', ')', ';', 'count', '+=', '1']},
    {'tokens': ['my_list', '=', '[', '1', ',', '2', ',', '3', ']', ';', 'my_list.append', '(', '4', ')']},
    {'tokens': ['my_dict', '=', '{', '"key1"', ':', '10', ',', '"key2"', ':', '20', '}', ';', 'print', '(', 'my_dict', '[', '"key1"', ']', ')']},
    {'tokens': ['import', 'math', ';', 'area', '=', 'math.pi', '*', 'radius', '**', '2']},
    {'tokens': ['class', 'MyClass', ':', 'def', '__init__', '(', 'self', ',', 'value', ')', ':', 'self.value', '=', 'value']},
    {'tokens': ['try', ':', 'divisor', '=', '0', ';', 'result', '=', '10', '/', 'divisor', ';', 'except', 'ZeroDivisionError', ':', 'print', '(', '"Cannot divide by zero"', ')']},
    {'tokens': ['function', 'greet', '(', 'name', ')', '{', 'console.log', '(', '"Hello, "', '+', 'name', ')', ';', '}']},
    {'tokens': ['let', 'data', '=', '[', '1', ',', '2', ',', '3', ']', ';', 'data.forEach', '(', 'item', '=>', 'console.log', '(', 'item', ')', ')', ';']},
    {'tokens': ['const', 'is_active', '=', 'true', ';', 'if', '(', 'is_active', ')', '{', 'console.log', '(', '"Active"', ')', ';}' , 'else', '{', 'console.log', '(', '"Inactive"', ')', ';}' ]},
    {'tokens': ['for', '(', 'let', 'j', '=', '0', ';', 'j', '<', '4', ';', 'j', '++', ')', '{', 'console.log', '(', 'j', ')', ';', '}']},
    {'tokens': ['public', 'class', 'Main', '{', 'public', 'static', 'void', 'main', '(', 'String', '[', ']', 'args', ')', '{', 'System.out.println', '(', '"Hello Java!"', ')', ';', '}', '}']},
    {'tokens': ['int', 'sum_java', '=', '0', ';', 'for', '(', 'int', 'k', '=', '0', ';', 'k', '<', '5', ';', 'k', '++', ')', '{', 'sum_java', '+=', 'k', ';', '}']},
    {'tokens': ['#include', '<iostream>', ';', 'int', 'main', '(', ')', '{', 'std::cout', '<<', '"Hello C++"', '<<', 'std::endl', ';', '}']},
    {'tokens': ['void', 'print_num', '(', 'int', 'num', ')', '{', 'std::cout', '<<', 'num', '<<', 'std::endl', ';', '}']},
    {'tokens': ['def', 'ruby_greet', '(', 'name', ')', ';', 'puts', '"Hello, #{name}"', ';', 'end']},
    {'tokens': ['5.times', '{', 'puts', '"Ruby Loop"', '}', ';']},
    {'tokens': ['package', 'main', ';', 'import', '"fmt"', ';', 'func', 'main', '(', ')', '{', 'fmt.Println', '(', '"Hello Go!"', ')', ';', '}']},
    {'tokens': ['func', 'add_go', '(', 'a', ',', 'b', 'int', ')', 'int', '{', 'return', 'a', '+', 'b', ';', '}']},
    {'tokens': ['SELECT', '*', 'FROM', 'users', 'WHERE', 'age', '>', '30', ';']},
    {'tokens': ['body', '{', 'font-family', ':', 'Arial', ',', 'sans-serif', ';', 'margin', ':', '0', ';', '}']}
]

dummy_tokenized_data = []
for item in dummy_tokenized_data_raw:
    normalized_tokens = normalize_tokens(item['tokens'])
    dummy_tokenized_data.append({'tokens': normalized_tokens})

# From cell ab45eb2d (Data splitting)
random.seed(42)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
random.shuffle(dummy_tokenized_data)
total_samples = len(dummy_tokenized_data)
train_split_idx = int(total_samples * train_ratio)
val_split_idx = int(total_samples * (train_ratio + val_ratio))
train_data = dummy_tokenized_data[:train_split_idx]
val_data = dummy_tokenized_data[train_split_idx:val_split_idx]
test_data = dummy_tokenized_data[val_split_idx:]

# From cell 1f91f2bc (Vocabulary building)
all_tokens = []
for item in train_data + val_data + test_data:
    all_tokens.extend(item['tokens'])
unique_tokens = sorted(list(set(all_tokens)))
special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
unique_tokens = [token for token in unique_tokens if token not in special_tokens]
vocab = {}
vocab['<PAD>'] = 0
for i, token in enumerate([t for t in special_tokens if t != '<PAD>']):
    vocab[token] = i + 1
current_id = len(special_tokens)
for token in unique_tokens:
    vocab[token] = current_id
    current_id += 1
id_to_token = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)
bos_token_id = vocab['<BOS>']
eos_token_id = vocab['<EOS>']
pad_token_id = vocab['<PAD>']
unk_token_id = vocab['<UNK>']

# From cell e7095f30 (Model hyperparameters)
d_model = 256
nhead = 8
num_layers = 6
dropout = 0.1
max_sequence_length = 512

model = TransformerDecoderModel(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout,
    max_sequence_length=max_sequence_length
)
# Instantiate custom datasets
train_dataset = CodeDataset(train_data, vocab, max_sequence_length, bos_token_id, eos_token_id, pad_token_id, unk_token_id)
val_dataset = CodeDataset(val_data, vocab, max_sequence_length, bos_token_id, eos_token_id, pad_token_id, unk_token_id)
test_dataset = CodeDataset(test_data, vocab, max_sequence_length, bos_token_id, eos_token_id, pad_token_id, unk_token_id)

# Define batch size
batch_size = 32

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
initial_lr = 0.0001 # Re-using previously defined initial learning rate
optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Ensure model is on the correct device
num_epochs = 10 # Re-using previously defined number of epochs

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Define the number of accumulation steps
accumulation_steps = 4 # Example: Accumulate gradients over 4 mini-batches

# Scaler for mixed precision training
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

# --- Demonstrating usage with and without copy mechanism ---
print("\n--- Demonstrating Code Generation with and without Copy Mechanism ---")

example_prompt_for_copy = ['def', 'my_function', '(', 'param1', ',', 'param2', ')', ':']

# Scenario 1: With Copy Mechanism Enabled
_, generated_code_with_copy = generate_code_sequence(
    model=model, vocab=vocab, id_to_token=id_to_token, max_sequence_length=max_sequence_length,
    device=device, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    unk_token_id=unk_token_id, start_prompt_tokens=example_prompt_for_copy,
    temperature=1.0, top_k=5, # Using sampling for more diverse comparison
    copy_mechanism_enabled=True, copy_score_boost=0.7
)
print(f"\nGenerated Code (with Copy Mechanism, boost=0.7):\nPrompt: {' '.join(example_prompt_for_copy)}\nGenerated: {generated_code_with_copy}")

# Scenario 2: Without Copy Mechanism
_, generated_code_no_copy = generate_code_sequence(
    model=model, vocab=vocab, id_to_token=id_to_token, max_sequence_length=max_sequence_length,
    device=device, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    unk_token_id=unk_token_id, start_prompt_tokens=example_prompt_for_copy,
    temperature=1.0, top_k=5, # Same sampling for fair comparison
    copy_mechanism_enabled=False # Explicitly disable
)
print(f"\nGenerated Code (without Copy Mechanism):\nPrompt: {' '.join(example_prompt_for_copy)}\nGenerated: {generated_code_no_copy}")
