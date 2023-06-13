import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CityscapeDataSet(Dataset):
    def __init__(self,img_dir,mask_dir):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.num_class = 4
        self.mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 0,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 0,  # building
            12: 0,  # wall
            13: 0,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 0,  # pole
            18: 0,  # polegroup
            19: 0,  # traffic light
            20: 0,  # traffic sign
            21: 0,  # vegetation
            22: 0,  # terrain
            23: 2,  # sky
            24: 0,  # person
            25: 0,  # rider
            26: 3,  # car
            27: 0,  # truck
            28: 0,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 0,  # motorcycle
            33: 0,  # bicycle
            -1: 0  # licenseplate
        }

    def __len__(self):
        return len(self.images)
    
    
    def class_to_mask(self,mask):
        mask_img = np.zeros_like(mask,dtype=np.uint8)
        for k in self.mapping:
            mask_img[mask==k] = self.mapping[k]
        return mask_img

    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.images[index].replace("leftImg8bit.png", "gtFine_labelIds.png"))
        
        image = Image.open(image_path).convert("RGB").resize((512,256))
        mask  = Image.open(mask_path).resize((512,256),Image.NEAREST)
        
        image = np.array(image,dtype=np.float16)/255.0
        mask = np.array(mask)

        image = torch.from_numpy(image).permute(2,0,1)
        mask = self.class_to_mask(mask)

        return image, mask


