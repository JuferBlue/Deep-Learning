import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../CIFAR10_data", download=True,train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]],dtype=torch.float32)
#
# input = torch.reshape(input,(1,1,5,5))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

net = Net()
# output = net(input)
# print(output)
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input_pool",imgs,step)
    output = net(imgs)
    writer.add_images("output_pool",output,step)
    step = step + 1

writer.close()