from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#forcing all images to be the same size (in this case 128x128 pixels)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=transform,
    download=True
)

test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    transform=transform,
    download=True
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)