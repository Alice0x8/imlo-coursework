from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#forcing all images to be the same size (in this case 128x128 pixels)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), #vonverting images into PyTorch Tensors
    transforms.Normalize((0.5,), (0.5,)) #shifting and scaling values to around [-1,1]
])

#loads the Oxford-IIIT Pet dataset
train_data = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=transform,
    download=True
)

#loads the test data
test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    transform=transform,
    download=True
)

#improving the gernealisation by splitting into batches of size 32 and shuffling them
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#creating test batches wihtout shuffling them to consistently compare accuracy in tests
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#takes one batch from training to verify
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)