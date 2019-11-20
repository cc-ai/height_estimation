import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvReluBN(nn.Module):
    """
    ConvReluBN for single conv relu BN branch of inception (conv 1x1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvReluBN, self).__init__()
        
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.BatchNorm2d(out_channels, affine=False),
        nn.ReLU(True),
        ))
    def forward(self, x):
        return self.conv[0](x)
    
class ConvReluBNbranch(nn.Module):
    """
    Conv Relu BN  for inception branch with 1x1 conv added
    """
    def __init__(self, in_channels, inter_dim, out_channels, kernel_size,  **kwargs):
        super(ConvReluBNbranch, self).__init__()
        self.conv = nn.ModuleList()
        
        # Base 1*1 conv layer
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels, inter_dim,1),
            nn.BatchNorm2d(inter_dim,affine=False),
            nn.ReLU(True),
            nn.Conv2d(inter_dim, out_channels, kernel_size, padding = int((kernel_size - 1)/2), **kwargs),
            nn.BatchNorm2d(out_channels,affine=False),
            nn.ReLU(True),
        ))
               
    def forward(self, x):
        return self.conv[0](x)
    
class InceptionS(nn.Module):
    """
    Inception module with 1x1, 3x3, 5x5, 7x7 size filters
    """
    def __init__(self, in_channels, inter_dim, out_ch):
        super(InceptionS, self).__init__()
        
        #in each branch, #out_channels is a quarter of the #ouput of the block because we concatenate the branche
        out_channels = int(out_ch/4)
        self.conv1x1 = ConvReluBN(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = ConvReluBNbranch(
            in_channels, inter_dim, out_channels, kernel_size=3)
        self.conv5x5 = ConvReluBNbranch(
            in_channels, inter_dim, out_channels, kernel_size=5)
        self.conv7x7 = ConvReluBNbranch(
            in_channels, inter_dim, out_channels, kernel_size=7)

    def forward(self, x):
        branch1x1 = self.conv1x1(x)
        branch3x3 = self.conv3x3(x)
        branch5x5 = self.conv5x5(x)
        branch7x7 = self.conv7x7(x)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], dim=1)

class InceptionL(nn.Module):
    """
    Inception module with 1x1, 3x3, 7x7, 11x11 size filters
    """
    def __init__(self, in_channels, inter_dim, out_ch):
        super(InceptionL, self).__init__()
        #in each branch, #out_channels is a quarter of the #ouput of the block because we concatenate the branches
        out_channels = int(out_ch/4)
        self.conv1x1 = ConvReluBN(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = ConvReluBNbranch(
            in_channels, inter_dim, out_channels, kernel_size=3)
        self.conv7x7 = ConvReluBNbranch(
            in_channels, inter_dim, out_channels, kernel_size=7)
        self.conv11x11 = ConvReluBNbranch(
            in_channels, inter_dim, out_channels, kernel_size=11)

    def forward(self, x): 
        branch1x1 = self.conv1x1(x)
        branch3x3 = self.conv3x3(x)
        branch7x7 = self.conv7x7(x)
        branch11x11 = self.conv11x11(x)
        return torch.cat([branch1x1, branch3x3, branch7x7, branch11x11], dim=1)

if __name__ == "__main__":
    
    x = Variable(torch.rand(2,128,125,125))
    testL = InceptionL(128,  64, 64)
    testS = InceptionS(128, 32, 128)
    print(testS(x))
    print(testL(x))
    