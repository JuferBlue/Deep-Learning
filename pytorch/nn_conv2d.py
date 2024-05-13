import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../CIFAR10_data", download=True,train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)


    def forward(self,x):
        x = self.conv1(x)
        return x

net = Net()
# print(net)
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,targets = data
    # print(imgs.shape)
    output = net(imgs)
    # print(output.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
    # 6个通道在tensorboard里报错
    #torch.Size([64, 6, 30, 30])->[xxx,3,30,30]
    #不知道写多少写-1，会自动计算
    output=torch.reshape(output,(-1,3,30,30))
    # print(output.shape)
    writer.add_images("output",output,step)
    step=step+1

