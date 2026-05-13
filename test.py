import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load test data
test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    transform=transform,
    download=True
)

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Evaluation
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