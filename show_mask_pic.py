import pickle
import os
import numpy as np
from PIL import Image
import torch

masks_dir = "./checkpoint/vgg16_7e-5alpha_lr005/best_keepratio_mask.pkl"

with open(masks_dir, "rb") as file:
    all_mask_values = pickle.load(file)  # dic
#keys = list(all_mask_values.keys())  # 'vgg_16/conv1/conv1_1/weights:0', 'vgg_16/conv1/conv1_2/weights:0'...
#print(all_mask_values)
# keylen = len(keys)

# key = keys[7]
'''
for i in keys:
    print(all_mask_values[i].shape)#(3, 3, 3, 64)     1
                                   #(3, 3, 64, 64)    2
                                   #(3, 3, 64, 128)   3
                                   #(3, 3, 128, 128)  4
                                   #(3, 3, 128, 256)  5
                                   #(3, 3, 256, 256)  6
                                   #(3, 3, 256, 256)  7
                                   #(3, 3, 256, 512)  8
                                   #(3, 3, 512, 512)  9
                                   #(3, 3, 512, 512)  10
                                   #(3, 3, 512, 512)  11
                                   #(3, 3, 512, 512)  12
                                   #(3, 3, 512, 512)  13


print((all_mask_values[key]).shape)
'''

##jiang yi ceng de mask bian wei er wei#########
count=0
cout = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
for kv in all_mask_values.items():
    #value = all_mask_values[key]
    value = kv[1]
    ratio = torch.sum(value) / value.numel()
    ratio = ratio.float()
    pic_w = cout[count]
    value = value.repeat(pic_w, 1, 1, 1)
    shape = value.shape
    print(shape)
    '''
    pic_h = shape[3]*shape[1]*shape[2]
    v = value.reshape(pic_w,pic_h).t()

    new_v = np.zeros((pic_h,pic_w,3))

    for i in range(pic_h):
        for j in range(pic_w):
            if not v[i][j]:
                new_v[i][j] = np.array([255,255,255])#False = bai, jian diao de shi bai se 
    new_v = new_v.astype(np.uint8)
    '''

    '''
    text='\n'.join(str(i) for i in v)
    with open("see_mask_bw.txt", "w") as fw:
        fw.write(text)
    '''

    '''
    new_pdf = Image.fromarray(new_v)
    count = count+1
    new_pdf.save("./vgg16_mask_pic_with_ratio/pic_vgg16_conv"+str(count)+"_{:.4f}.pdf".format(ratio))
    '''
    count = count+1
    if count == 13:
        break

