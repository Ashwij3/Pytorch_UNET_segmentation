import cv2
import os
import numpy as np
import torch
from model import UNet

def main():
    img = cv2.imread("/home/ak47/Downloads/cityscape/train_images/aachen_000000_000019_leftImg8bit.png")
    img = cv2.resize(img,(512,256))
    print(img.shape)
    cv2.imshow("test",img)
    
    img_array = np.asarray(img,dtype=np.float32)/255
    img_array = np.expand_dims(img_array, axis=0)
    data = torch.from_numpy(img_array).permute(0,3,1,2).to(device=DEVICE)
    print(data.shape)


    current_dir = os.getcwd()
    model = UNet(in_channels=3,out_channels=4).to(DEVICE)
    weight_path = os.path.join(current_dir,"weights/checkpoint_75.pth.tar")

    weight = torch.load(weight_path)
    model.load_state_dict(weight["state_dict"])

    model.eval()

    out = model(data)
    print(out.shape)

    output = torch.argmax(torch.squeeze(out),0).cpu().numpy()
    print(output.shape)

    output[np.where(output==0)] = 50
    output[np.where(output==1)] = 100
    output[np.where(output==2)] = 150
    output[np.where(output==3)] = 200

    output = np.array(output,dtype=np.uint8)

    cv2.imshow("Output",output)

    cv2.waitKey(0)    


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main()