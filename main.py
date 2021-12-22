# utility functions
import model # model implementation
import trainer # training functions
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

    # initialize the model or loading the pre-train model
    if opt.pretrained:
        model_train = torch.load(opt.pretrained_path) 
    else:
        model_train = model.prepare_model(opt)
    
    # configurate the optimizer and learning rate scheduler
    optim, sche = optimizer.prepare_optim(model_train, opt)

    # train the model
    model_train = trainer.train(model_train, optim, sche, opt)

    # save the final trained model
    utils.save_model(model_train, opt)
    
    return 

if __name__ == '__main__':
    main()
