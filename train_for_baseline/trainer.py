import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging
import os
from tensorboardX import SummaryWriter
import utils
import datetime
import os
import getData
import random

def train(model, optim, sche, opt):
    """
    Args:
        model: the model to be trained
        optim: pytorch optimizer to be used
        opt: command line input from the user
    """

    # save the training log and the loss graph
    now_time = datetime.datetime.now()
    logname = opt.save_log + str(now_time) + "log.txt"
    writer = SummaryWriter(opt.save_loss + str(now_time) + '/')
    
    # make dirction
    if not os.path.exists(opt.save_loss):
        os.makedirs(opt.save_loss)
    if not os.path.exists(opt.save_log):
        os.makedirs(opt.save_log)
    
    # evaluate model
    psnr_max = evaluate(logname,model,0, opt)

    # set loss function
    loss_function = nn.MSELoss(reduction='mean')
        
    # start training
    for epoch in range(1, opt.epochs + 1):

        # set the model in the training mode
        model.train()

        # shuffle the .h5 files and train the model
        hdf5_Idx = random.sample(range(0, opt.train_len), opt.train_len)
        for h5idx in range(opt.train_len):

            # data loader
            train_set = getData.TrainSet(hdf5_Idx[h5idx], opt)
            train_loader = torch.utils.data.DataLoader(train_set, 
                                                    batch_size = opt.batch_size, 
                                                    shuffle = True)    

            for batch_idx, batch in enumerate(train_loader):

                data = batch
                if opt.cuda:
                    with torch.no_grad():
                        # move to GPU
                        data = data.cuda()               

                # erase all computed gradient        
                optim.zero_grad()
                
                # forward pass to get prediction
                hsi_pred = model(data)

                loss=loss_function(hsi_pred, data)

                # save the loss info
                writer.add_scalar("loss",loss, (epoch * opt.train_len + h5idx) * opt.batch_num + batch_idx)

                # compute gradient in the computational graph
                loss.backward(retain_graph=False)
                
                # update parameters in the model 
                optim.step()

                # update learning rate
                sche.step()

                # logging
                if batch_idx % opt.report_every == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                        epoch, batch_idx * opt.batch_size, len(train_set),
                        100. * batch_idx / len(train_loader), loss.data.item()))

        # evaluate model
        psnr_tmp = evaluate(logname,model,epoch, opt)
        if psnr_tmp > psnr_max:
            utils.save_model(model, opt)
            psnr_max = psnr_tmp

    logging.info('Training finished.')
    writer.close()
    return model

def evaluate(logname,model,epoch, opt):
    
    # set the model in the evaluating mode
    model.eval()
    if opt.cuda:
        model.cuda()

    PSNR = 0.
    psnr_cache = []

    # test one .h5 file by one
    for h5idx in range(opt.valid_len):

        # data loader
        eval_set = getData.ValidSet(h5idx, opt)
        loader = torch.utils.data.DataLoader(eval_set, 
                                            batch_size = opt.batch_size, 
                                            shuffle=False, )

        for batch_idx, batch in enumerate(loader):

            data = batch
            with torch.no_grad():
                if opt.cuda:
                    data = data.cuda()             
                hsi_pred = model(data) 
                data,hsi_pred = data.cpu(), hsi_pred.cpu()
                hsi_pred = np.array(hsi_pred)
                data = np.array(data)
                #calculate psnr
                for i in range(len(data)):
                    psnr_cache.append(utils.Cal_PSNR_by_gt(data[i],hsi_pred[i]))

    # calculate the average PSNR of the total validset
    PSNR = sum(psnr_cache)/len(psnr_cache)

    # record the results
    logging.info(' Average_PSNR: {:.4f}. '.format(PSNR))  
    f = open(logname,'a')
    f.write('Epoch:')
    f.write(str(str(epoch)))
    f.write('      ')
    f.write(str(PSNR))
    f.write('\r\n')
    f.close

    return PSNR