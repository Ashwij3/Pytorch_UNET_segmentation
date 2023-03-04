import torch
import torchvision.transforms.functional as TF
from .UNet_parts import *

class UNet(torch.nn.Module):
    def __init__(self,in_channels, out_channels,features=[64,128,256,512]):
        super().__init__()
        self.down = torch.nn.ModuleList()
        self.up = torch.nn.ModuleList()
        
        self.bottle_neck = DoubleConv(in_channels=features[-1], out_channels=2*features[-1])
        self.output = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        for feature in features:
            self.down.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        

        for feature in reversed(features):
            self.up.append(torch.nn.ConvTranspose2d(feature*2,feature,kernel_size=2, stride=2))
            self.up.append(DoubleConv(feature*2,feature))
        

    
    def forward(self,x):
        skip_connections = []


        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = torch.nn.MaxPool2d(kernel_size=2)(x)
            
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1] 

        for i in range(0,len(self.up),2):
            x = self.up[i](x)

            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up[i+1](concat_skip)

        x = self.output(x)
        return x


def test():
    x = torch.randn((3, 1, 572, 572))
    model = Unet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()

