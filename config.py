import torch

# Hyperparameters and Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
VOCAB_SIZE = None  # Set during tokenization
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_CLASSES = None  # Set dynamically
NUM_HEADS = 8
LEARNING_RATE = 0.001