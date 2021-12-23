import torch

# set the optimizer and the scheduler
def prepare_optim(model, opt):
    params = [ p for p in model.parameters() if p.requires_grad]

    # optimizer type
    if opt.optim_type == 'adam':
        optimizer = torch.optim.Adam(params, lr = opt.lr, 
                                     weight_decay = opt.weight_decay)
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr = opt.lr, 
                                    momentum = opt.momentum,
                                    weight_decay = opt.weight_decay)   

    # scheduler with pre-defined learning rate decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones = opt.milestones * opt.batch_size * opt.train_len, 
                                                    gamma = opt.gamma)

    return optimizer, scheduler