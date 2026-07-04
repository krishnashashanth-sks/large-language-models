def get_training_corpus(dataset):
    return (
        dataset[split][i]["text"]
        for split in ["train", "validation"]
        for i in range(len(dataset[split]))
        if dataset[split][i]["text"]
    )

# Function to concatenate all texts and then chunk them into fixed block_size
def group_texts(examples):
    block_size = 128 # A common block size for language models
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop and a tricky mask.
    # The default currently is to drop information if the last batch is smaller than block_size.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result