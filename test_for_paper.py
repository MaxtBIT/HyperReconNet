import os
import numpy as np
import torch
import utils
import cv2
import train_for_mask_optimization.model as model

# parse command line input
opt = utils.parse_arg()

# set parameters
dataPath = opt.testset_path
model_path = opt.pretrained_path
channel = opt.channel
patch_len = opt.patch_size
stride = int(opt.patch_size / 2)

# Set GPU
opt.cuda = opt.gpuid>=0
if opt.cuda:
    torch.cuda.set_device(opt.gpuid)
else:
    raise NotImplementedError

# save the test results for each image
psnr_list = []
ssim_list = []
sam_list = []

# load the model
model = model.prepare_model(opt)
model.load_state_dict(torch.load(model_path,  map_location='cuda:0') )

# get the list of testset
pics = os.listdir(dataPath)
for pic_name in pics:
    input_path = dataPath + pic_name + '/'
    print(pic_name)

    # extract data
    data_list = []
    frames = os.listdir(input_path)
    frames.sort(key=lambda x:int(x[:-4].split('-')[1]))
    for frame in frames:
        data_list.append(np.expand_dims(cv2.imread(input_path + frame,-1), 2))
    img = np.concatenate(data_list, 2) / 255.0
    gt = img

    #roll the  image to be tested
    img_roll = np.zeros(img.shape, dtype='float32')
    for ch in range(channel):
        img_roll[:,:,ch] = np.roll(img[:, :, ch], shift = -ch, axis = 0)

    # divide into patches
    patch_list = []
    row_max = (img_roll.shape[0] - patch_len)//stride
    col_max = (img_roll.shape[1] - patch_len)//stride
    for i in range(row_max + 1):    #row
        if i < row_max :
            for j in range(col_max + 1):    #col
                if j < col_max :
                    patch_list.append(img_roll[ stride * i : stride * i + patch_len, stride * j : stride * j + patch_len, :])
                else:
                    patch_list.append(img_roll[ stride * i : stride * i + patch_len, img_roll.shape[1] - patch_len : img_roll.shape[1], :])
        else:
            for j in range(col_max + 1):    #col
                if j < col_max :
                    patch_list.append(img_roll[ img_roll.shape[0] - patch_len : img_roll.shape[0], stride * j : stride * j + patch_len, :])
                else:
                    patch_list.append(img_roll[ img_roll.shape[0] - patch_len : img_roll.shape[0], img_roll.shape[1] - patch_len : img_roll.shape[1], :])

    input_data = torch.FloatTensor(patch_list)

    # model interface
    pred_list = []
    model.eval()
    if opt.cuda:
        model.cuda()

    loader = torch.utils.data.DataLoader(input_data, 
                                            batch_size = opt.batch_size, 
                                            shuffle=False, )

    for batch_idx, batch in enumerate(loader):
        data = batch
        with torch.no_grad():
            if opt.cuda:
                data = data.cuda()             
            hsi_pred = model(data) 
            hsi_pred = hsi_pred.cpu()
            pred_list.append(np.array(hsi_pred))

    # get the prediction
    predict_list = np.concatenate(pred_list, 0)

    # joint the patches and remove the image boundaries
    row_list = []
    for i in range(row_max + 1):    #row
        tmp = col_max//2
        row_odd = predict_list[ i * (col_max + 1)]
        for j in range(tmp - 1):
            row_odd = np.concatenate((row_odd, predict_list[ i * (col_max + 1) + 2 * (j + 1)]), 1)
        row_even = predict_list[ i * (col_max + 1) + 1]
        for j in range(tmp - 2):
            row_even = np.concatenate((row_even, predict_list[ i * (col_max + 1) + 2 * (j + 1) + 1]), 1)
        row_data = (row_odd[:, stride : row_odd.shape[1] - stride, :] + row_even)/2
        row_list.append(row_data)

    half_row_list = []
    col_num = len(row_list) - 1
    for i in range(col_num - 1):
        col_data = (row_list[i][stride::] + row_list[i + 1][0:stride])/2
        half_row_list.append(col_data)

    predict_img = np.concatenate(half_row_list, 0)

    #unroll
    output = np.zeros(predict_img.shape, dtype='float32')
    for ch in range(channel):
        output[:,:,ch] = np.roll(predict_img[:, :, ch], shift = ch, axis = 0)

    # crop the rolled boundaries and keep the same size for both images
    output = output[stride : output.shape[0] - stride, :, :]
    gt = gt[stride*2 : stride*2 + output.shape[0], stride : stride + output.shape[1], :]

    # get the center of the image
    gt_cenrow = gt.shape[0] //2
    gt_cencol = gt.shape[1] //2

    # cut out a central square with a side length of 512 for comparison
    output = output[gt_cenrow//2 - (stride*2 +16) : gt_cenrow//2 - (stride*2 +16) + 512, gt_cencol//2 - stride : gt_cencol//2 - stride + 512, :]
    gt = gt[gt_cenrow//2 - (stride*2 +16) : gt_cenrow//2 - (stride*2 +16) + 512, gt_cencol//2 - stride : gt_cencol//2 - stride + 512, :]

    gt = gt.astype('float32')
    output = output.astype('float32')

    # calculate the psnr
    PSNR = utils.Cal_PSNR_by_gt(gt, output)
    psnr_list.append(PSNR)
    print('PSNR:' + str(PSNR))

    #calculate the ssim
    SSIM = utils.Cal_SSIM(gt, output)
    ssim_list.append(SSIM)
    print('SSIM:' + str(SSIM))

    #calculate the sam
    SAM = utils.Cal_SAM(gt, output)
    sam_list.append(SAM)
    print('SAM:' + str(SAM))

# calculate the average results
psnr_avg = sum(psnr_list)/len(psnr_list)
print('psnr_average:  ' + str(psnr_avg))

ssim_avg = sum(ssim_list)/len(ssim_list)
print('ssim_average:  ' + str(ssim_avg))

sam_avg = sum(sam_list)/len(sam_list)
print('sam_average:  ' + str(sam_avg))