import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="../CIFAR10_data",train=True,download=True,transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="../CIFAR10_data",train=False,download=True,transform=dataset_transform)

# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)

# print(test_set[0])

writer = SummaryWriter(log_dir="logs")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()