import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import CNN

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)

# -----------------------------
# Transforms (simple + stable)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset (SIMPLE version)
# -----------------------------
full_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    transform=transform,
    download=True
)

val_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    transform=val_transform,
    download=True
)

# -----------------------------
# Train / Val split (simple random_split)
# -----------------------------
num_samples = len(full_dataset)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, _ = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

_, val_dataset = random_split(
    val_dataset,
    [train_size, val_size],
    generator=generator
)

# -----------------------------
# Loaders
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Model
# -----------------------------
model = CNN().to(device)

# -----------------------------
# Loss + Optimiser
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)

# -----------------------------
# Training
# -----------------------------
epochs = 30
best_val = 0.0

for epoch in range(epochs):
    model.train()

    train_correct = 0
    train_total = 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "model.pth")
        print("Saved best model!")

print("Training complete.")
print("Best Val:", best_val)