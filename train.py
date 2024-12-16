import torch
import torch.optim as optim
import torch.nn as nn
import config
from models.models import get_baseline_model, get_attention_model
from data.preprocess import load_data

def train_model(model, train_loader, val_loader, epochs=config.EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs = torch.stack(batch['input_ids']).to(config.DEVICE)
            labels = torch.stack(batch['input_ids']).to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.argmax(dim=1)  # Assuming labels are one-hot encoded
            loss = criterion(outputs, labels.squeeze())

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.stack(batch['input_ids']).to(config.DEVICE)
                labels = torch.stack(batch['input_ids']).to(config.DEVICE)
                outputs = model(inputs)
                labels = labels.argmax(dim=1)

                val_loss = criterion(outputs, labels.squeeze())
                total_val_loss += val_loss.item()

        # Average validation loss for the epoch
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, GAM-RHN - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    torch.save(model.state_dict(), 'models/save_state/baseline_model.pth')
    print(f"Model saved to models/save_state/baseline_model.pth")

def train_model_attn(model, train_loader, val_loader, epochs=config.EPOCHS, model_name="dot_product"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs = torch.stack(batch['input_ids']).to(config.DEVICE)
            labels = torch.stack(batch['input_ids']).to(config.DEVICE)

            optimizer.zero_grad()
            outputs, _ = model(inputs)  # Unpack outputs and ignore attention weights for loss calculation
            labels = labels.argmax(dim=1)  # Assuming labels are one-hot encoded
            loss = criterion(outputs, labels.squeeze())

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.stack(batch['input_ids']).to(config.DEVICE)
                labels = torch.stack(batch['input_ids']).to(config.DEVICE)
                outputs, _ = model(inputs)
                labels = labels.argmax(dim=1)

                val_loss = criterion(outputs, labels.squeeze())
                total_val_loss += val_loss.item()

        # Average validation loss for the epoch
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, {model_name} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    torch.save(model.state_dict(), f'models/save_state/{model_name}_model.pth')
    print(f"Model saved to models/save_state/{model_name}_model.pth")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()
    name = input("\n\nWhich model to train? (baseline/dot_product/scaled_dot_product/additive/multi_head): ")
    model = get_baseline_model() if name == "baseline" else get_attention_model(name)
    train_model(model, train_loader, val_loader) if name == "baseline" else train_model_attn(model, train_loader, val_loader, name)