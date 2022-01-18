import torch
import torch.nn as nn
from train_for_mask_optimization.modules import *
import utils

class CodedNet(nn.Module):

    def __init__(self):

        super(CodedNet, self).__init__()

        opt = utils.parse_arg()
        self.patch_size = opt.patch_size
        self.channel = opt.channel
        self.mask_size = int(opt.patch_size / 2)   #32 * 32
        self.weight_size = int(opt.patch_size * opt.patch_size / 4) # 32 * 32 = 1024

        # set the BinaryNet
        self.bin_conv = nn.Sequential(
            BinaryConv2d(self.weight_size, self.weight_size, kernel_size=1, padding=0, groups=self.weight_size),)

    def forward(self, x):               #[B, 64, 64, 31]

        batch_size = x.size(0)      #batchsize大小
        measure_input = torch.zeros([batch_size, self.patch_size, self.patch_size, self.channel])

        # roll x
        for ch in range(self.channel):
            measure_input[:, :, :, ch] = torch.roll(x[:, :, :, ch], shifts = -ch, dims = 1)

        #divide one sample to 4 parts
        y1 = measure_input[:, 0:self.mask_size, 0:self.mask_size, :].cuda().reshape(batch_size, self.weight_size, self.channel).unsqueeze(2)
        y2 = measure_input[:, self.mask_size:self.patch_size, 0:self.mask_size, :].cuda().reshape(batch_size, self.weight_size, self.channel).unsqueeze(2)
        y3 = measure_input[:, 0:self.mask_size, self.mask_size:self.patch_size, :].cuda().reshape(batch_size, self.weight_size, self.channel).unsqueeze(2)
        y4 = measure_input[:, self.mask_size:self.patch_size, self.mask_size:self.patch_size, :].cuda().reshape(batch_size, self.weight_size, self.channel).unsqueeze(2)
        
        #put them through the same conv layer
        y1 = self.bin_conv(y1).reshape(batch_size, self.mask_size, self.mask_size, self.channel)
        y2 = self.bin_conv(y2).reshape(batch_size, self.mask_size, self.mask_size, self.channel)
        y3 = self.bin_conv(y3).reshape(batch_size, self.mask_size, self.mask_size, self.channel)
        y4 = self.bin_conv(y4).reshape(batch_size, self.mask_size, self.mask_size, self.channel)

        #cat 4 parts into one sample
        tmp1 = torch.cat((y1,y2),  1)
        tmp2 = torch.cat((y3,y4),  1)
        x = torch.cat((tmp1,tmp2),  2)

        # unroll x
        for ch in range(self.channel):
            measure_input[:, :, :, ch] = torch.roll(x[:, :, :, ch], shifts = ch, dims = 1)
        
        # product the measurement
        measure_input = torch.sum(measure_input, 3)                                      #[B, 64, 64]

        return measure_input

def CodedModel():
    return CodedNet()
 
