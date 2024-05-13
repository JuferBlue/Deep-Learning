from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "../practice_data/train/ants_image/24335309_c5ea483bb8.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("toTensor",img_tensor)



#nomalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6,3,2],[9,3,5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)
#``output[channel] = (input[channel] - mean[channel]) / std[channel]``

#resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(img_resize.shape)
writer.add_image("Resize",img_resize,0)

#Compose resize 等比缩放
#compose 执行流水作业
trans_resize_2 = transforms.Resize(512)
#先resize在toTensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)




#randomcrop随机裁剪

trans_random = transforms.RandomCrop(50)

trans_compose_2 = transforms.Compose([trans_random,trans_totensor])

for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()



