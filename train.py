import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

from model import CNN

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)

# -----------------------------
# Transforms
# -----------------------------
# Training transform includes data augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation transform has NO augmentation
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load full dataset twice
# -----------------------------
# One copy with augmentation for training
full_train_aug = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    transform=train_transform,
    download=True
)

# One copy without augmentation for validation
full_train_no_aug = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    transform=val_transform,
    download=True
)

# -----------------------------
# Train / Validation split
# -----------------------------
num_samples = len(full_train_aug)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

# Use fixed seed so the split is reproducible
generator = torch.Generator().manual_seed(42)
train_split, val_split = random_split(
    range(num_samples),
    [train_size, val_size],
    generator=generator
)

# Extract indices from splits
train_indices = train_split.indices
val_indices = val_split.indices

# Create subsets with different transforms
train_dataset = Subset(full_train_aug, train_indices)
val_dataset = Subset(full_train_no_aug, val_indices)

# -----------------------------
# Data loaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# -----------------------------
# Model
# -----------------------------
model = CNN().to(device)

# -----------------------------
# Loss function and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)

# Learning-rate scheduler:
# reduce LR when validation accuracy stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# -----------------------------
# Training settings
# -----------------------------
epochs = 30
best_val_accuracy = 0.0

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):
    # ----- Training -----
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item()

        # Training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    # ----- Validation -----
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    # Adjust learning rate based on validation accuracy
    scheduler.step(val_accuracy)

    # Print progress
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save best model (based on validation accuracy)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "model.pth")
        print("Saved new best model!")

# -----------------------------
# Training finished
# -----------------------------
print("Training complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")