import torch
import torch.nn as nn
import config
from data.preprocess import load_data
from models.models import get_baseline_model, get_attention_model

def evaluate_model(model, val_loader, model_path='models/save_state/baseline_model.pth'):
    # Load the pretrained model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    perplexity_loss = 0
    total_tokens = 0
    correct = 0
    total = 0
    batch_count = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in val_loader:
            inputs = torch.stack(batch['input_ids']).to(config.DEVICE)
            labels = torch.stack(batch['input_ids']).to(config.DEVICE)

            outputs = model(inputs)  # Get model outputs
            labels = labels.argmax(dim=1)  # Match the labels as in the training function

            # Calculate loss
            loss = criterion(outputs, labels.squeeze())
            total_loss += loss.item()
            perplexity_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, -1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store batch-wise accuracy and perplexity
            accuracy_batch = (predicted == labels).sum().item() / labels.size(0)
            perplexity_batch = torch.exp(torch.tensor(loss.item()))  # Perplexity for the current batch

            print(f'Batch {batch_count} - Accuracy: {accuracy_batch*100:.2f}%, Perplexity: {perplexity_batch:.4f}')

            batch_count += 1

    # Compute accuracy and perplexity
    accuracy = correct / total
    perplexity = torch.exp(torch.tensor(perplexity_loss / total_tokens))

    print(f'Validation Loss: {total_loss/len(val_loader)}, Accuracy: {accuracy*100:.2f}%, Perplexity: {perplexity:.4f}')

def evaluate_model_attn(model, val_loader, name='dot_product'):
    # Load the pretrained model
    model.load_state_dict(torch.load(f'models/save_state/{name}_model.pth'))
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    perplexity_loss = 0
    total_tokens = 0
    correct = 0
    total = 0
    batch_count = 0
    criterion = nn.CrossEntropyLoss()

    accuracies_batch = []  # List to store accuracy for each batch
    perplexities_batch = []  # List to store perplexity for each batch

    with torch.no_grad():  # Disable gradient calculation for evaluation
        batch_count = 0
        for batch in val_loader:
            # Extract inputs and move to the device
            inputs = torch.stack(batch['input_ids']).to(config.DEVICE)
            labels = torch.stack(batch['input_ids']).to(config.DEVICE)

            outputs, _ = model(inputs)  # Unpack outputs and attention weights, use only outputs (logits)
            labels = labels.argmax(dim=1)  # Match the labels as in the training function

            # Calculate loss
            loss = criterion(outputs, labels.squeeze())
            total_loss += loss.item()
            perplexity_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, dim=1)  # Get predicted class
            correct += (predicted == labels).sum().item()  # Compare predictions to true labels
            total += labels.size(0)

            # Store batch-wise accuracy and perplexity
            accuracy_batch = (predicted == labels).sum().item() / labels.size(0)
            perplexity_batch = torch.exp(torch.tensor(loss.item()))  # Perplexity for the current batch

            print(f'Batch {batch_count} - Accuracy: {accuracy_batch*100:.2f}%, Perplexity: {perplexity_batch:.4f}')

            batch_count += 1

    # Compute overall accuracy and perplexity
    accuracy = correct / total  # Overall accuracy
    perplexity = torch.exp(torch.tensor(perplexity_loss / total_tokens))  # Overall perplexity

    print(f'Validation Loss: {total_loss/len(val_loader)}, '
          f'Accuracy: {accuracy*100:.2f}%, Perplexity: {perplexity:.4f}')

if __name__ == "__main__":
    _, _, test_loader = load_data()
    name = input("\n\nWhich model to evaluate? (baseline/dot_product/scaled_dot_product/additive/multi_head): ")
    model = get_baseline_model() if name == "baseline" else get_attention_model(name)
    evaluate_model(model, test_loader) if name == "baseline" else evaluate_model_attn(model, test_loader, name)