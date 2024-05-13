import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
#
# input = torch.reshape(input,(-1,1,2,2))
# print(input.shape)
dataset = torchvision.datasets.CIFAR10(root="../CIFAR10_data", download=True,train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output


net = Net()
# output = net(input)
# print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input_sig",imgs,step)
    output = net(imgs)
    writer.add_images("output_sig",output,step)
    step = step + 1

writer.close()