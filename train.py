import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import SimpleCNN, train_dataset, train_loader, test_dataset, test_loader
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/pokemon_training_2")

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=len(train_dataset.classes)).to(device)
print(f'Training on {device}')
# Loss and Optimizer
# Count class samples in your dataset
class_counts = Counter(train_dataset.labels)
total_samples = sum(class_counts.values())
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
class_weights = torch.tensor(class_weights, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log batch loss to TensorBoard
        writer.add_scalar('Training Loss/Batch', loss.item(), epoch * len(train_loader) + batch_idx)

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Log epoch loss to TensorBoard
    writer.add_scalar('Training Loss/Epoch', epoch_loss, epoch)


# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Log test accuracy to TensorBoard
writer.add_scalar('Test Accuracy', accuracy, epoch)

# Log the model architecture
sample_image = torch.randn(1, 3, 128, 128).to(device)
writer.add_graph(model, sample_image)

writer.close()


# Save the model
torch.save(model.state_dict(), "pokemon_cnn_model.pth")
