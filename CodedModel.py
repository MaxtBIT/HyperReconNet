import torch
import torch.nn as nn
from modules import *

class CodedNet(nn.Module):

    def __init__(self):

        super(CodedNet, self).__init__()
        
        # set the BinaryNet
        self.bin_conv = nn.Sequential(
            BinaryConv2d(1024, 1024, kernel_size=1, padding=0, groups=1024),)

    def forward(self, x):               #[B, 64, 64, 31]

        batch_size = x.size(0)      #batchsize大小
        measure_input = torch.zeros([batch_size, 64, 64, 31])

        # roll x
        for ch in range(31):
            measure_input[:, :, :, ch] = torch.roll(x[:, :, :, ch], shifts = -ch, dims = 1)

        #divide one sample to 4 parts
        y1 = measure_input[:, 0:32, 0:32, :].cuda().reshape(batch_size, 1024, 31).unsqueeze(2)
        y2 = measure_input[:, 32:64, 0:32, :].cuda().reshape(batch_size, 1024, 31).unsqueeze(2)
        y3 = measure_input[:, 0:32, 32:64, :].cuda().reshape(batch_size, 1024, 31).unsqueeze(2)
        y4 = measure_input[:, 32:64, 32:64, :].cuda().reshape(batch_size, 1024, 31).unsqueeze(2)
        
        #put them through the same conv layer
        y1 = self.bin_conv(y1).reshape(batch_size, 32, 32, 31)
        y2 = self.bin_conv(y2).reshape(batch_size, 32, 32, 31)
        y3 = self.bin_conv(y3).reshape(batch_size, 32, 32, 31)
        y4 = self.bin_conv(y4).reshape(batch_size, 32, 32, 31)

        #cat 4 parts into one sample
        tmp1 = torch.cat((y1,y2),  1)
        tmp2 = torch.cat((y3,y4),  1)
        x = torch.cat((tmp1,tmp2),  2)

        # unroll x
        for ch in range(31):
            measure_input[:, :, :, ch] = torch.roll(x[:, :, :, ch], shifts = ch, dims = 1)
        
        # product the measurement
        measure_input = torch.sum(measure_input, 3)                                      #[B, 64, 64]

        return measure_input

def CodedModel():
    return CodedNet()
 
