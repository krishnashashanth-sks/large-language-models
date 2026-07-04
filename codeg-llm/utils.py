import torch,re
def tokens_to_code(token_ids, id_to_token_map, special_token_ids):
    """
    Converts a list of token IDs back to a human-readable code string.
    Filters out special tokens like <BOS>, <EOS>, <PAD>.
    """
    code_tokens = []
    for token_id in token_ids:
        if token_id not in special_token_ids:
            token = id_to_token_map.get(token_id, '<UNK>') # Use <UNK> for any unmapped ID
            code_tokens.append(token)

    # Basic joining logic - can be refined for better code formatting if needed
    # For now, join with spaces and handle common punctuation without leading spaces
    code_string = ' '.join(code_tokens)
    code_string = code_string.replace(' .', '.')
    code_string = code_string.replace(' ,', ',')
    code_string = code_string.replace(' ;', ';')
    code_string = code_string.replace(' :', ':')
    code_string = code_string.replace(' ( ', '(')
    code_string = code_string.replace(' )', ')')
    code_string = code_string.replace(' [ ', '[')
    code_string = code_string.replace(' ]', ']')
    code_string = code_string.replace(' { ', '{')
    code_string = code_string.replace(' }', '}')
    code_string = code_string.replace(' != ', '!=') # Example for operators
    code_string = code_string.replace(' == ', '==')
    code_string = code_string.replace(' <= ', '<=')
    code_string = code_string.replace(' >= ', '>=')
    code_string = code_string.replace(' = ', '=')
    code_string = code_string.replace(' + ', '+')
    code_string = code_string.replace(' - ', '-')
    code_string = code_string.replace(' * ', '*')
    code_string = code_string.replace(' / ', '/')
    code_string = code_string.replace(' += ', '+=')

    # Remove multiple spaces that might result from replacements
    code_string = ' '.join(code_string.split())

    return code_string

def normalize_tokens(tokens):
    """
    Normalizes a list of tokens by standardizing literals and handling whitespace.
    """
    normalized = []
    for token in tokens:
        token_str = str(token).strip().lower()
        if not token_str: continue
        if re.match(r'^-?\d+(\.\d+)?$', token_str): normalized.append('<NUM_LITERAL>'); continue
        if (token_str.startswith('"') and token_str.endswith('"')) or \
           (token_str.startswith('\'') and token_str.endswith('\'')): normalized.append('<STR_LITERAL>'); continue
        normalized.append(token_str)
    return normalized

# This function would be defined earlier in cell fa7db296
def generate_causal_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask