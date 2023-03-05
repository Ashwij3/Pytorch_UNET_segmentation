import torch
from .dataset import BasicDataSet
from torch.utils.data import DataLoader

def get_loaders(
    train_img_dir,
    train_mask_dir,
    valid_img_dir,
    valid_mask_dir,
    batch_size,
    num_workers,
    pin_memory):

    train_Dataset = BasicDataSet(train_img_dir, train_mask_dir)
    train_loader = DataLoader(train_Dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)

    valid_Dataset = BasicDataSet(valid_img_dir, valid_mask_dir)
    valid_loader = DataLoader(valid_Dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)

    return train_loader,valid_loader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
