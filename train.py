import torch
import torchvision
from torchvision import transforms
from backbone.EfficientFace import *
from NRE.model import *

model = NREModel()

root = r'C:\Users\Alex\Documents\Uni_Data\faces\fer2013\train'

train_set = torchvision.datasets.ImageFolder(root, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(train_set, batch_size=32)

for i, (x, y) in enumerate(trainloader):
    x = torch.rand((32, 3, 224, 224))
    p, domain_prob = model(x)
    print(p.shape)
    break
