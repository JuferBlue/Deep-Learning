import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="../CIFAR10_data", train=False, download=True,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,num_workers=0,batch_size=64,shuffle=False,drop_last=True)

img,target = test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter(log_dir="logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step= step+1

writer.close()
