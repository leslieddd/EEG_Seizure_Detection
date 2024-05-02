# import torch
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader
# from dataset2 import EEGDataset  # Assuming EEGDataset is correctly implemented
# from model2 import EEGSNet  # Assuming EEGSNet is correctly implemented
#
# # Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Initialize dataset
# data_dir = '/Users/sizheng/Desktop/EEG_Seizure_Detection-main/sleep_data_sample/'
# subj_nums = [i for i in range(1, 3)]  # Example subject numbers
# dataset = EEGDataset(subj_nums=subj_nums, root_path=data_dir)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# # Initialize model
# model = EEGSNet().to(device)
# criterion = torch.nn.BCEWithLogitsLoss()  # Appropriate for binary classification
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training loop
# num_epochs = 50 # Number of epochs
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, (data, labels) in enumerate(train_loader):
#         data = data.to(device)
#         labels = labels.to(device).unsqueeze(1).float()  # Make sure labels are the correct shape
#
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#
#         # Forward + backward + optimize
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 10 == 9:  # Print every 10 mini-batches
#             print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10:.4f}')
#             running_loss = 0.0
#
# print('Finished Training')
#
# # Save the trained model
# torch.save(model.state_dict(), 'model_state_dict.pth')
#
# # Optionally, save the entire model
# torch.save(model, 'model_entire.pth')
#
# # Assume a testing dataset is prepared similarly
# test_dataset = EEGDataset(subj_nums=[33], root_path=data_dir)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
#
# # Function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(data)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Apply threshold
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy
#
# # Call the evaluate function after training
# accuracy = evaluate_model(model, test_loader, device)
#
#
#
# # Export hyperparameters and other metadata
# hyperparameters = {
#     'learning_rate': 0.001,
#     'batch_size': 10,
#     'num_epochs': num_epochs,
#     'model_architecture': 'EEGSNet',
#     'criterion': 'BCEWithLogitsLoss',
#     'optimizer': 'Adam',
#     'accuracy': accuracy
# }
#
# import json
# with open('model_hyperparameters.json', 'w') as f:
#     json.dump(hyperparameters, f)

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from dataset2 import EEGDataset  # Assuming EEGDataset is correctly implemented
from model2 import EEGSNet  # Assuming EEGSNet is correctly implemented

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Initialize dataset
data_dir = '/Users/sizheng/Desktop/EEG_Seizure_Detection-main/sleep_data_sample/'
subj_nums = [i for i in range(1, 2)]  # Example subject numbers
print("Loading dataset...")
dataset = EEGDataset(subj_nums=subj_nums, root_path=data_dir)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset loaded and DataLoader prepared.")

# Initialize model
model = EEGSNet().to(device)
print("Model initialized and moved to device.")
criterion = torch.nn.BCEWithLogitsLoss()  # Appropriate for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Optimizer and criterion set up.")

# Training loop
num_epochs = 20 # Number of epochs
print("Starting training loop...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device).unsqueeze(1).float()  # Make sure labels are the correct shape
        print(f'Epoch {epoch+1}, Batch {i+1}: Data loaded to device.')

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'model_state_dict.pth')
print('Model state_dict saved.')

torch.save(model, 'model_entire.pth')
print('Entire model saved.')

# Assume a testing dataset is prepared similarly
print("Preparing test dataset...")
test_dataset = EEGDataset(subj_nums=[2], root_path=data_dir)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
print("Test dataset prepared. Starting evaluation...")

# Call the evaluate function after training
accuracy = evaluate_model(model, test_loader, device)
print(f'Training completed with Test Accuracy: {accuracy * 100:.2f}%')

# Export hyperparameters and other metadata
hyperparameters = {
    'learning_rate': 0.001,
    'batch_size': 10,
    'num_epochs': num_epochs,
    'model_architecture': 'EEGSNet',
    'criterion': 'BCEWithLogitsLoss',
    'optimizer': 'Adam',
    'accuracy': accuracy
}

import json
with open('model_hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)
print("Hyperparameters and other metadata saved to JSON.")

