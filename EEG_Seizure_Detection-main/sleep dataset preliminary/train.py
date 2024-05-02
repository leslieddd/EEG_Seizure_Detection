import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from dataset import EEGDataset  # Assuming EEGDataset is correctly implemented
from model import EEGSNet  # Assuming EEGSNet is correctly implemented
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

# Set device for training (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

# Hyperparameters
learning_rate = 0.01
batch_size = 16
num_epochs = 10

# Model, loss, and optimizer
model = EEGSNet().to(device)
state_dict = torch.load('model_checkpoint.pth')
model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()  # Appropriate for classification tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data directories
root_dir = '/Users/jeonsang-eon/sleep_data_processed/'



def collate_fn(batch):
    # Sort the batch in the order of decreasing sequence length
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)

    # Calculate the maximum sequence length in this batch
    max_seq_len = max([s.size(0) for s in sequences])

    # Pad sequences to the max length found in the batch
    padded_sequences = torch.stack([
        F.pad(seq, (0, 0, 0, 0, 0, 0, 0, max_seq_len - seq.size(0)))  # Only pad the sequence dimension
        for seq in sequences
    ], dim=0)


    # Convert list of labels tensors to a single tensor
    labels = torch.stack(labels, dim=0)  # Stack the labels which are assumed to be tensors already
    labels = torch.argmax(labels, dim=1)  # Convert from one-hot to indices if they are one-hot encoded

    # Calculate the original lengths of each sequence for potential use in models like RNN
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    return padded_sequences, labels, lengths




def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, lengths in train_loader:
            assert labels.dtype == torch.long, "Labels must be of type torch.long"
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Ensure outputs are logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')



def test_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print(f'Test Loss: {total_loss / len(test_loader.dataset):.4f}')
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Training dataset
train_nums = [1,2,3,4]
train_dataset = EEGDataset(subj_nums=train_nums, root_path=root_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Testing dataset
test_nums = [5]
test_dataset = EEGDataset(subj_nums=test_nums, root_path=root_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Test the model
test_model(model, test_loader, criterion)

# Save the model checkpoint
torch.save(model.state_dict(), 'model_checkpoint.pth')
print("Model saved to model_checkpoint.pth")

