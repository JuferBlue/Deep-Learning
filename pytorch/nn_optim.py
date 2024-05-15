import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="../CIFAR10_data", download=True,train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3,32,5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)

        self.model1 =   Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )


    def forward(self,x):
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
net = Net()

optim = torch.optim.SGD(net.parameters(),lr=0.1,)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        outputs = net(imgs)
        loss_result = loss(outputs,targets)
        optim.zero_grad()

        loss_result.backward()

        optim.step()
        # print(loss_result)
        # print("ok")
        running_loss += loss_result
    print(running_loss)