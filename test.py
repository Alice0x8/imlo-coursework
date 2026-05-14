import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# uses GPU if possible, otherwise uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defines the transformation of the images (same as training data)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] # matches the training data
    )
])

# loads the Oxford IIIT Pet dataset
test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",   
    transform=transform,
    download=True
)

# loads the required data for the test
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval() # uses evaluation mode to prevent training specific behaviour

# tracks number of correct predictions and total samples
correct = 0
total = 0

# test loop
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# prints final test accuracy
print(f"Test Accuracy: {100 * correct / total:.2f}%")