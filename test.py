
import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

from train_argument import parser, print_args

from time import time
from utils import *
from models import *
from trainer import *


#w = nn.Parameter(torch.ones(6, 3, 3, 3))
#w = nn.Parameter(torch.ones(4,2))
#w = nn.Parameter(torch.Tensor([[1,0,0,1,0],[1,0,0,0,0]]))
#w = w.reshape([5,2])
#print(w)
#wv = w.repeat(1,2)
#print(wv)
#re = nn.Parameter(torch.zeros(2, 3, 4, 2))
#nn.init.uniform_(w, 0, 1)
#w_view = w.view(w.shape[0], -1)
#t = nn.Parameter(torch.Tensor([2.]))
#print("t:",t)
#q = torch.sum(w.data, dim=0)#/float(w_view.shape[0]) #- t
#print(q)  #[3,3,3]
#nn.init.uniform_(t, 0, 1)
#t_view = t.view(w.shape[0],-1)
'''
msk=[[[1.,1.],
      [1.,1.],
      [1.,0.],
      [1.,1.]],

     [[1.,0.],
      [1.,0.],
      [1.,1.],
      [1.,1.]],

     [[1.,1.],
      [1.,1.],
      [1.,1.],
      [1.,1.]]]
'''
#msk=[[[0.,1.,1.],[1.,1.,1.],[1.,1.,0.]],[[1.,0.,0.],[0.,0.,0.],[1.,1.,1.]],[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]
#msk = [1, 0, 0, 1]
#m = torch.Tensor(msk)
#mw = torch.einsum('ij,i->ij',w,m)
#print(mw)
#mm = m.repeat(64,1,1,1)
#mm_view = mm.view(mm.shape[0],-1)
#print("m:",torch.sum(m)/m.numel())
#print("mm:",torch.sum(mm_view)/mm_view.numel())
#masked_weight = torch.einsum('ij,i->ij',w_view,m)
#print("masked_weight:",type(masked_weight))
#m_veiw = m.view(m.shape[0],-1)
#print(m.shape)
#out = w.shape[0]
#for i in range(out):
#    re[i] = w[i]*m
#print(re)
#torch.sum(w_view.data, dim=1)
#print(w_view.shape)
#print(w_view*m_veiw)
#print(t_view)
#re = w_view-t_view
#print(re)

file_name="./checkpoint/vgg16_5e-3alpha_lr02_bn64/best_acc_model.pth"

net = masked_vgg(dataset="cifar10", depth=16)

net.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))
'''查看网络结构
for i in net.modules():
      for j in i.feature:
            if isinstance(j, MaskedConv2d):
                  print(j.weight[0])   #是定值
                  break
      
      break
'''
input_data = torch.randn(64,3,32,32,dtype=torch.float).cuda()
net.cuda()
starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions,1))
for i in range(10):
      a = net(input_data)
with torch.no_grad():
      for rep in range(repetitions):
            starter.record()
            out = net(input_data)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

mean_syn = np.sum(timings)/repetitions
print(mean_syn)
