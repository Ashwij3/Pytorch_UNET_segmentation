import torch


class DoubleConv(torch.nn.Module):
    def __init__(self,in_channels, out_channels, mid_channels = None) -> None:
        super().__init__()
        if mid_channels == None:
            mid_channels = out_channels
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2)
        )
    
    def forward(self,x):
        return self.model(x)
