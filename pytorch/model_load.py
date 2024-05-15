import torch
import torchvision.models

#保存方式1加载模型
model = torch.load('vgg16_method1.pth')
# print(model)

#方式2
vgg16 = torchvision.models.vgg16(pretrained=False)

model_2 = torch.load('vgg16_method2.pth')

vgg16.load_state_dict(model_2)
# print(vgg16)


#test陷阱

model_test = torch.load("test.pth")
print(model_test)