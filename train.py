import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from utils.dataset import CityscapeDataSet
from utils import get_loaders,save_checkpoint
from model import UNet


def train(loader,model,loss_func,optimizer):
    model.train()
    loop = tqdm(loader)
    step = 0
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        step+=1
        
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE,dtype=torch.long)

        
        with torch.cuda.amp.autocast():
            pred = model(data)
            loss = loss_func(pred, targets)
            total_loss +=loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        loop.set_postfix(loss=total_loss/step)
    
    return total_loss/step


def main():
    current_dir = os.getcwd()
    with open(os.path.join(current_dir,'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader,valid_loader = get_loaders(CityscapeDataSet,
                                            config['TRAIN_IMG_DIR'],
                                            config['TRAIN_MASK_DIR'],
                                            config['VAL_IMG_DIR'],
                                            config['VAL_MASK_DIR'],
                                            config['BATCH_SIZE'],
                                            config['NUM_WORKERS'],
                                            config['PIN_MEMORY'])

    
    model = UNet(in_channels=3,out_channels=config["num_classes"]).to(DEVICE)
    if config['LOAD_MODEL']:
        weight = torch.load(os.path.join(current_dir,"checkpoint_5.pth.tar"))
        model.load_state_dict(weight["state_dict"])
    
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['LEARNING_RATE'],weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250)
    
    
    for epoch in range(config['NUM_EPOCHS']):     
        print("EPOCH: ",epoch)  
        epoch_loss = train(train_loader,model,loss_func,optimizer)

        if epoch%15==0 and epoch!=0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch_loss": epoch_loss
            }
            save_checkpoint(checkpoint,filename="weights/checkpoint_{}.pth.tar".format(epoch))

        lr_scheduler.step()






if __name__=='__main__':
    torch.cuda.empty_cache()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main()

