import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)


    def forward(self,x) :
        return self.conv1(x)

net = Net()
torch.save(net,"test.pth")
