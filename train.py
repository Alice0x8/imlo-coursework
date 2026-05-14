import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import CNN

# printing device to confirm it is training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#random starting point to split the data into train and validation groups
torch.manual_seed(42)

# transformation section
transform = transforms.Compose([
    transforms.Resize((128, 128)), # standardises image size for the CNN
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3), # 3 rotations gave me the highest training value
    transforms.ToTensor(), # converts image to Pytorch tensore format
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# transformation section for the validation set (no augmentatoion)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# loading the Oxford IIIT Pet dataset
dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    transform=transform,
    download=True
)

# Splitting the dataset into training and validation set (in this case using 80/20)
num_samples = len(dataset)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

#again setting a fixed starting point to maintain the split
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

# loading the data to allow them to be batched and shuffled
# training data is shuffled to improve learning 
# valiation data is not shuffled so it can be consistently compared
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# initialising the CNN
model = CNN().to(device)

# loss and optimiser:
criterion = nn.CrossEntropyLoss() # used as there is 37 classes

# using the Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005, # the learning rate (step size)
    weight_decay=1e-4 # trying to stop it from overfitting
)

# this reduces the learning rate every 10 epochs by scale of 0.5 to help stabilise the convergence 
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

# Training:
epochs = 30
best_val = 0.0

for epoch in range(epochs):
    model.train()

    # to track the training performance
    train_correct = 0
    train_total = 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        #cmoputing the loss
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # computes accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # gets the predicted class label 
            _, predicted = torch.max(outputs, 1)

            # tracks validation accuracy
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # computes the accuracy
    val_acc = 100 * val_correct / val_total

    # prints the progress for each epoch
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

    # saves the best validation accuracy
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "model.pth")
        print("Saved best model!")

    # updates learning rate set earlier in schuedler
    scheduler.step()

# prints best validation value
print("Training complete.")
print("Best Validation:", best_val)