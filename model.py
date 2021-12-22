import torch.nn as nn
import CodedModel
import ReconModel

# set the entire structure of the network
class HyperReconNet(nn.Module):
    def __init__(self, codednet, reconnet):

        super(HyperReconNet, self).__init__()
        self.codednet = codednet
        self.reconnet = reconnet
        
    def forward(self, x):

        measure_pic = self.codednet(x) #Coded
        Output_hsi = self.reconnet(measure_pic) #Reconstruction

        return Output_hsi

def prepare_model(opt):
    # HyperReconNet consists of two parts:
    #1. the coded aperture optimization
    #2. the HSI reconstruction
    codedmodel = CodedModel.CodedModel()
    reconmodel = ReconModel.ReconModel()   
    model = HyperReconNet(codedmodel, reconmodel)     
    
    if opt.cuda:
        model = model.cuda()
    else:
        raise NotImplementedError

    return model