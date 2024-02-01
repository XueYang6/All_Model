from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.data[idx]['label'])
        return img, label


class ResNetClassification(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassification, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
