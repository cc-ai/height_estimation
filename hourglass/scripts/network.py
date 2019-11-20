import torch
from torch import nn
from torch.autograd import Variable
from inception_module import InceptionS, InceptionL 
from torch import optim

class Channels1(nn.Module):
    def __init__(self):
        super(Channels1, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                InceptionS(256, 32, 256),
                InceptionS(256,32, 256)
                )
            ) #EE
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                InceptionS(256,32, 256),
                InceptionS(256,32,256),
                InceptionS(256,32,256), 
                nn.UpsamplingNearest2d(scale_factor=2)
                )
            )
         #EEE

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)

class Channels2(nn.Module):
    def __init__(self):
        super(Channels2, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                InceptionS(256,32,256), 
                InceptionL(256,64,256)
                )
            )#EF
        self.list.append( 
            nn.Sequential(
                nn.AvgPool2d(2),
                InceptionS(256,32,256), 
                InceptionS(256,32,256), 
                Channels1(),
                InceptionS(256,32,256), 
                InceptionL(256,64,256),
                nn.UpsamplingNearest2d(scale_factor=2)
            )#EE1EF
        )
    def forward(self,x):
        return self.list[0](x)+self.list[1](x)

            
class Channels3(nn.Module):
    def __init__(self):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                InceptionS(128,32,128), 
                InceptionS(128,32,256),
                Channels2(),
                InceptionS(256,32,256),
                InceptionS(256,32,128), 
                nn.UpsamplingNearest2d(scale_factor=2)
               )#BD2EG
            )
        self.list.append(
            nn.Sequential(
                InceptionS(128,32,128), 
                InceptionL(128, 64, 128)
                )
            )#BC                    

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)

class Channels4(nn.Module):
    def __init__(self):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                InceptionS(128,32,128),
                InceptionS(128,32,128),
                Channels3(),
                InceptionS(128,32,128),
                InceptionL(128,32,64),
                nn.UpsamplingNearest2d(scale_factor=2)
                )
            )#BB3BA
        self.list.append(
            nn.Sequential(
                InceptionL(128,32,64)
                )
            )#A

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3,128,7,padding=3), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            Channels4(),
            nn.Conv2d(64,1,3,padding=1)
            )

    def forward(self,x):
        # print(x.data.size())
        return self.seq(x)

def get_model():
    return Model().cuda()

if __name__ == "__main__":
    cuda0 = torch.device('cuda:0')
    test = Model().to(device=cuda0)
    print(test)
    x = torch.tensor(torch.rand(1,3,256,256)).to(device=cuda0)
    print(test(x))