import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BasicDataSet(Dataset):
    def __init__(self,img_dir,mask_dir):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.images[index].replace(".jpg", ".png"))
        
        image = np.array(Image.open(image_path).convert("RGB").resize((512,256)))
        mask = np.array(Image.open(mask_path).convert("L").resize((512,256)),dtype=np.float32)

        assert image.shape[:2] == mask.shape
        return image, mask


# d = BasicDataSet("/home/ak47/unet/UNET_segmentatioin/data/train_images","/home/ak47/unet/UNET_segmentatioin/data/train_masks")
# d.__getitem__(index)
