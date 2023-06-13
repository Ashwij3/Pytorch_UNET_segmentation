import os
import yaml
import torch
import time
import torchvision
from tqdm import tqdm
from model import UNet
import torchvision.utils as v_utils
from utils import get_loaders
from utils.dataset import CarvanaDataSet, CityscapeDataSet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def eval(loader,model,ouput_folder):

    result_path  = os.path.join(ouput_folder,"results")
    history_accuracy = []
    history_time = []
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):

    # send to the GPU and do a forward pass
        start_time = time.time()
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        pred = model(data)
        end_time = time.time()

        pred_class = torch.zeros((pred.size()[0], pred.size()[2], pred.size()[3])).to(device=DEVICE)
        for idx in range(0, pred.size()[0]):
            pred_class[idx] = torch.argmax(pred[idx], dim=0)
            
            

    # # we "squeeze" the groundtruth if we are using cross-entropy loss
    # # this is because it expects to have a [N, W, H] image where the values
    # # in the 2D image correspond to the class that that pixel should be 0 < pix[u,v] < classes
    # if args.losstype == "segment":

    # # max over the classes should be the prediction
    # # our prediction is [N, classes, W, H]
    # # so we max over the second dimension and take the max response
    # # if we are doing rgb reconstruction, then just directly save it to file
    # pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
    # if args.losstype == "segment":
    #     for idx in range(0, y.size()[0]):
    #         pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()
    #         #pred_rgb[idx] = img_data.class_to_rgb(maxindex)
    # else:
    #     print("this test script only works for \"segment\" unet classification...")
    #     exit()

    # # unsqueese so we have [N, 1, W, H] size
    # # this allows for debug saving of the images to file...
    pred_class = pred_class.unsqueeze(1).float()
    label_class = targets.unsqueeze(1).float()

    # # now compare the groundtruth to the predicted
    # # we should record the accuracy for the class
    acc_sum = (pred_class == label_class).sum()
    acc = float(acc_sum) / (label_class.size()[0]*label_class.size()[2]*label_class.size()[3])
    history_accuracy.append(acc)
    history_time.append((end_time-start_time))

    # debug saving generated classes to file
    v_utils.save_image(pred_class.float()/13, os.path.join(result_path,("gen_image_{}_{}.png".format(0, batch_idx))))
    v_utils.save_image(label_class.float()/13,os.path.join(result_path,("label_image_{}_{}.png".format(0, batch_idx))) )
    v_utils.save_image(x.cpu().data,os.path.join(result_path,("original_image_{}_{}.png".format(0, batch_idx))))


# finally output the accuracy
    print("\nNETWORK RESULTS")
    print("    - avg timing = %.4f (sec)" % (sum(history_time)/len(history_time)))
    print("    - avg accuracy = %.4f" % (sum(history_accuracy)/len(history_accuracy)))
#     data ,targets = next(iter(loader))
#     print(data.shape)
#     data = (data.permute(0,3,1,2)).float().to(DEVICE)
#     preds = torch.sigmoid(model(data))
#     preds = (preds > 0.5).float()
#     torchvision.utils.save_image(
#             preds, os.path.join(ouput_folder,"pred_1.png")
#         )

def main():
    current_dir = os.getcwd()
    with open(os.path.join(current_dir,'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    model = UNet(in_channels=3,out_channels=config["num_classes"]).to(DEVICE)
    weight_path = os.path.join(current_dir,"checkpoint_26.pth.tar")
    print(weight_path)
    weight = torch.load(weight_path)
    model.load_state_dict(weight["state_dict"])


    train_loader,valid_loader = get_loaders(CityscapeDataSet,
                                            config['TRAIN_IMG_DIR'],
                                            config['TRAIN_MASK_DIR'],
                                            config['VAL_IMG_DIR'],
                                            config['VAL_MASK_DIR'],
                                            config['BATCH_SIZE'],
                                            config['NUM_WORKERS'],
                                            config['PIN_MEMORY'])

    model.eval()
    eval(train_loader,model,current_dir)


if __name__=="__main__":
    main()