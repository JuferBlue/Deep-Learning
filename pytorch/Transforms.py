from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
#用法  tensor数据类型

img_path = "../practice_data/train/ants_image/24335309_c5ea483bb8.jpg"

img = Image.open(img_path)

write = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

write.add_image("Tensor_img",tensor_img)
write.close()