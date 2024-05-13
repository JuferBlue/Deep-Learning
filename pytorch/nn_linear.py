import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../CIFAR10_data", download=True,train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608,10)


    def forward(self,input):
        output = self.linear1(input)
        return output

net = Net()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,targets = data
    # writer.add_images("input_pool",imgs,step)
    # output = net(imgs)
    # writer.add_images("output_pool",output,step)

    output = torch.flatten(imgs)
    output = net(output)

    step = step + 1

writer.close()
