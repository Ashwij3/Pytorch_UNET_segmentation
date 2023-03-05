import os
import torch
import yaml
from tqdm import tqdm
from utils import get_loaders,save_checkpoint
from model import UNet


torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(loader,model,loss_func,optimizer,scaler):
    model.train()
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = (data.permute(0,3,1,2)).float()
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        with torch.cuda.amp.autocast():
            pred = model(data)
            loss = loss_func(pred, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    current_dir = os.getcwd()
    with open(os.path.join(current_dir,'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader,valid_loader = get_loaders(config['TRAIN_IMG_DIR'],
                                            config['TRAIN_MASK_DIR'],
                                            config['VAL_IMG_DIR'],
                                            config['VAL_MASK_DIR'],
                                            config['BATCH_SIZE'],
                                            config['NUM_WORKERS'],
                                            config['PIN_MEMORY'])

    model = UNet(in_channels=3,out_channels=1).to(DEVICE)
    loss_func = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['LEARNING_RATE'])
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.1)
    
    if config['LOAD_MODEL']:
        weight = torch.load(os.path.join(current_dir,"pre_trained.pth.tar"))
        model.load_state_dict(weight["state_dict"])
    
    for epoch in range(config['NUM_EPOCHS']):
        train(train_loader,model,loss_func,optimizer,scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        save_checkpoint(checkpoint,filename="checkpoint_{}.pth.tar".format(epoch))

        # lr_scheduler.step()


        # with torch.cuda.amp.autocast():
        #     pred = model(data)
        #     loss = loss_func(pred, targets)
        #     print(loss.item())
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()





if __name__=='__main__':
    main()

