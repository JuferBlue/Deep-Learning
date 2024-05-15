import torchvision
from torch import nn
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
# print("ok")
# print(vgg16_false)

dataset = torchvision.datasets.CIFAR10(root="../CIFAR10_data", download=True,train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))

print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)