from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter("logs")
image_path = "../practice_data/train/ants_image/24335309_c5ea483bb8.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("train",img_array,1,dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

# writer.add_scalar()
writer.close()




