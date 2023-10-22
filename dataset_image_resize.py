from PIL import Image
import os
import sys
import torchvision.transforms as transforms

save_image_data = r'D:\minor_project_gan\datasets\celeba_renew'
get_image_data = r'D:\minor_project_gan\datasets\CelebAMask-HQ\CelebA-HQ-img'
new = r'D:\minor_project_gan\datasets\dataset for'
from PIL import Image


for i in range(5000):
    path = f'{save_image_data}\{i}.jpg'
    image = Image.open(path)
   # print(f"Original size : {image.size}")
    #image = image.resize((256,256))
    filepath = f'{new}\{i}.jpg'
    print(i)
   # print(f"Resized image : {image.size}")
    image.save(filepath)

#sunset_resized.save(filepath)