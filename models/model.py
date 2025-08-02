#importingn the necesseary libraries for resizing the images
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from collections import Counter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#resizing the images
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
#applying it
train_data = ImageFolder("data/chest_xray/train", transform=train_transform)
val_data = ImageFolder("data/chest_xray/val", transform=val_transform)
test_data = ImageFolder("data/chest_xray/test", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)


# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)

# Instantiate and move model to GPU
model = CNN().to(device)
print("Model created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# Use only this version for cleaner weight calculation
class_counts = Counter(train_data.targets)  
class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(3)], dtype=torch.float)
class_weights = class_weights / class_weights.sum()

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Accuracy function
def calculate_accuracy(loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Simplified for CrossEntropy output
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Training loop
n_epochs = 15
best_val_acc = 0.0
patience = 5
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    val_acc = calculate_accuracy(val_loader, model)
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        print(f" No improvement. Patience left: {patience - patience_counter}")
        if patience_counter >= patience:
            print(" Early stopping triggered.")
            break

    # Fail-safe early exit
    if train_loss > 2.0 and val_acc < 0.5:
        print(" Training loss too high and accuracy too low. Exiting early.")
        break

    #Pass validation accuracy to ReduceLROnPlateau
    scheduler.step(val_acc)

# Final accuracies
train_acc = calculate_accuracy(train_loader, model)
val_acc = calculate_accuracy(val_loader, model)
print(f" Final Train Accuracy: {train_acc:.4f}")
print(f" Final Val Accuracy: {val_acc:.4f}")
