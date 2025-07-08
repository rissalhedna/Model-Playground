from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 100 
train_data_loader = DataLoader(train, batch_size = batch_size)
test_data_loader = DataLoader(test, batch_size = batch_size)

