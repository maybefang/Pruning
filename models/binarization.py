import math
import torch 
import torch.nn as nn

"""
Function for activation binarization
"""
class BinaryStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional



#masked linear layer(fc): matrix shape is [outsize, insize]
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        #self.threshold = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(1))
        self.step = BinaryStep.apply
        self.mask = None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0)
    
    def forward(self, input):
        abs_weight = torch.abs(self.weight)
        #threshold = self.threshold.view(abs_weight.shape[0], -1)
        mean_weight = torch.sum(abs_weight.data, dim=1)/float(abs_weight.shape[1])
        if self.threshold.data<0:
            self.threshold.data.fill_(0.)
        mean_weight = mean_weight-self.threshold #[cin]
        mask = self.step(mean_weight) #[cin]
        ratio = torch.sum(mask) / mask.numel()
        ###########################################################
        #print("mlp keep ratio {:.2f}".format(ratio))
        #print("mlp threshold {:3f}".format(self.threshold[0]))
        ##########################################################
        if ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0)
            abs_weight = torch.abs(self.weight)
            #threshold = self.threshold.view(abs_weight.shape[0], -1)
            #threshold = self.threshold
            mean_weight = torch.sum(abs_weight.data, dim=1)/float(abs_weight.shape[1])
            mean_weight = mean_weight-self.threshold
            mask = self.step(mean_weight)
        self.mask = mask.repeat(abs_weight.shape[1], 1).t() #mask=[cin],self.mask=[cin,cout]
        #masked_weight = self.weight * mask
        masked_weight = torch.einsum('ij,i->ij', self.weight, mask)
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output

    


class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c 
        self.out_channels = out_c
        self.kernel_size = kernel_size   #(h,w)
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups
        self.mask = None

        ## define weight 
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        #set t, t is a number not a vector
        self.threshold = nn.Parameter(torch.Tensor(1))  #if out_c=3,init [0,0,0],out_c=2 init [0,0] the numbers are 0
        self.step = BinaryStep.apply
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def forward(self, x):
        weight_shape = self.weight.shape #[out,in,w,h]
        #threshold = self.threshold.view(weight_shape[0], -1)
        weight = torch.abs(self.weight)
        #weight = weight.view(weight_shape[0], -1)
        weight = torch.sum(weight.data, dim=0)/float(weight_shape[0])#weight.shape=[cin,w,h]
        if self.threshold.data<0:      # make t>=0
            self.threshold.data.fill_(0.)
        weight = weight - self.threshold
        mask = self.step(weight)                       #mask.shape=[cin,w,h]
        #mask = mask.view(weight_shape)
        ratio = torch.sum(mask) / mask.numel()
        ####################################################################
        #print("conv threshold {:3f}".format(self.threshold[0]))
        #print("conv keep ratio {:.2f}".format(ratio))
        ######################################################################
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.)
            #threshold = self.threshold.view(weight_shape[0], -1)
            weight = torch.abs(self.weight)
            #weight = weight.view(weight_shape[0], -1)
            weight = torch.sum(weight.data, dim=0) / float(weight_shape[0])
            weight = weight - self.threshold
            mask = self.step(weight)
            #mask = mask.view(weight_shape)
        #self.mask = mask    #save the [cin,w,h] size mask
        self.mask = mask.repeat(weight_shape[0],1,1,1)#old:[cin,w,h] new:[cout,cin,w,h]
        masked_weight = torch.einsum('ijkl,jkl->ijkl', self.weight, mask)

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out


    
