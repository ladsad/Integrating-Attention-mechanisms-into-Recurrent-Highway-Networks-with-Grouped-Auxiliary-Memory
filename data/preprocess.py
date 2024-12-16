import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import config

# Load and preprocess the PTB dataset
def load_data(batch_size=config.BATCH_SIZE):
    dataset = load_dataset('ptb_text_only')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=64)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset, val_dataset, test_dataset = tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    config.VOCAB_SIZE = tokenizer.vocab_size
    config.NUM_CLASSES = config.VOCAB_SIZE

    return train_loader, val_loader, test_loader
