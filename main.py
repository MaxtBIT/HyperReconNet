# utility functions
# train for baseline
import train_for_baseline.model as model_baseline # model implementation
import train_for_baseline.trainer as trainer_baseline # training functions
# train for learnable mask
import train_for_mask_optimization.model as model_optim # model implementation
import train_for_mask_optimization.trainer as trainer_optim # training functions
import optimizer # optimization functions
import utils # other utilities

# public libraries
import torch
import logging
import numpy as np
import time
import os

def main():

    # logging configuration
    logging.basicConfig(level = logging.INFO,
        format = "[%(asctime)s]: %(message)s"
    )
        
    # parse command line input
    opt = utils.parse_arg()

    # Set GPU
    opt.cuda = opt.gpuid>=0
    if opt.cuda:
        torch.cuda.set_device(opt.gpuid)
    else:
        raise NotImplementedError

    # record the current time
    opt.save_dir += time.asctime(time.localtime(time.time()))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # initialize the model
    if opt.mode == 'baseline':
        model_train = model_baseline.prepare_model(opt)
        # loading the pre-train model (if need)
        if opt.pretrained:
            model_train = torch.load(opt.pretrained_path,  map_location='cuda:0') 
    elif opt.mode == 'optim':
        model_train = model_optim.prepare_model(opt)
        # loading the pre-train model (if need)
        if opt.pretrained:
            model_train.load_state_dict(torch.load(opt.pretrained_path,  map_location='cuda:0') )
    else:
        raise NotImplementedError
    
    # configurate the optimizer and learning rate scheduler
    optim, sche = optimizer.prepare_optim(model_train, opt)

    # train the model
    if opt.mode == 'baseline':
        model_train = trainer_baseline.train(model_train, optim, sche, opt)
    elif opt.mode == 'optim':
        model_train = trainer_optim.train(model_train, optim, sche, opt)
    else:
        raise NotImplementedError

    # save the final trained model
    utils.save_model(model_train, opt)
    
    return 

if __name__ == '__main__':
    main()
