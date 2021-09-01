import torch
from torch import nn
# from torch.autograd import Variable
import torch.nn.functional as F
##Original CBAM##

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1, affine=True)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # x = self.bn(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        
        out = x * (self.ca(x))
        out = out * (self.sa(out))
        
        return out

## ZAM ##    
class ZeroChannelAttention(nn.Module):
    def __init__(self):
        super(ZeroChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.avg_pool(x) + self.max_pool(x))

class ZeroSpatialAttention(nn.Module):
    def __init__(self):
        super(ZeroSpatialAttention, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(avg_out + max_out)
    
class ZAM(nn.Module):
    def __init__(self, use_skip_connection = False):
        super(ZAM, self).__init__()

        self.ca = ZeroChannelAttention()
        self.sa = ZeroSpatialAttention()
        self.use_skip_connection = use_skip_connection
        
    def forward(self, x):
        
        out = x + x * self.ca(x) if self.use_skip_connection else x * self.ca(x)
        out = out + out * self.sa(out) if self.use_skip_connection else out * self.sa(out)
        
        return out

### my attention ###
class MyChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(MyChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=True)
        # self.fc3   = nn.Conv2d()
        self.sigmoid = nn.Sigmoid()
        self.wgt  = nn.Parameter(torch.Tensor([1,1]))
        

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        var_out = torch.var(x,dim=[2,3]).unsqueeze(2).unsqueeze(3)
        # var_out = self.fc2(self.relu1(self.fc1(var_out)))

        # avg_pow_out = self.avg_pool(x.pow(4)).pow(1/4)
        # var_out = self.fc2(self.relu1(self.fc1(avg_pow_out)))

        var_out = self.fc2(self.relu1(self.fc1(var_out)))
        # out = F.softmax(self.wgt)[0]*avg_out + F.softmax(self.wgt)[1]*var_out
        # out = self.sigmoid(self.wgt[0]) * avg_out + self.sigmoid(self.wgt[1]) * var_out
        out =  avg_out +  var_out
        # print(self.wgt)

        # out = F.softmax(self.wgt)[0]*avg_out + F.softmax(self.wgt)[1]*avg_pow_out
        # print(F.softmax(self.wgt),'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        return self.sigmoid(out)

class MySpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(MySpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        var_out = torch.var(x, dim=1, keepdim=True)
        # avg_pow_out = torch.mean(x.pow(2), dim=1, keepdim=True).sqrt()
        # x = torch.cat([avg_out, var_out], dim=1)
        x = torch.cat([avg_out, var_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MVAM(nn.Module):
    def __init__(self, in_planes):
        super(MVAM, self).__init__()

        self.ca = MyChannelAttention(in_planes)
        self.sa = MySpatialAttention()
        
    def forward(self, x):
        
        out = x * (self.ca(x))
        out = out * (self.sa(out))
        
        return out

class SSAM(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, num_key = 10):
        super(SSAM, self).__init__()
        self.chanel_in = in_dim
        self.key = num_key
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=self.key, kernel_size=1)
        self.linear1 = nn.Conv2d(in_channels = self.key, out_channels = 4*self.key, kernel_size=1, bias=False)
        self.linear2 = nn.Conv2d(in_channels = 4*self.key, out_channels = self.key, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W)
            returns :
                out : attention value + input feature
                attention: B x (HxW) x (HxW)
        """
        m_batchsize, C, height, width = x.size()
        out_map = self.conv(x)                     # prob_map: B x K x H x W
        wgt = self.conv.weight.detach()                 # wgt: K x C x 1 x 1

        # compute the factor from each kernel
        prob_map = out_map.detach()
        norm_map = torch.norm(prob_map, dim = 1)    # norm_map: B x H x W
        norm_wgt = torch.norm(wgt, dim = 1)         # norm_wgt: K x 1 x 1
        prob_map = prob_map/norm_wgt
        for i in range(m_batchsize):
            prob_map[i] = prob_map[i]/norm_map[i]

        ratio = self.sigmoid(self.linear2(F.relu(self.linear1(self.avg_pool(prob_map)))))
        out = 0
        for i in range(self.key):
            zz =  prob_map[:,i,:,:].unsqueeze(1).repeat(1,self.chanel_in,1,1) * wgt[i,:,:,:].unsqueeze(0) * ratio[:,i].unsqueeze(1).repeat(1,self.chanel_in,1,1)
            out = out + prob_map[:,i,:,:].unsqueeze(1).repeat(1,self.chanel_in,1,1) * wgt[i,:,:,:].unsqueeze(0) * ratio[:,i].unsqueeze(1).repeat(1,self.chanel_in,1,1)
        return self.gamma * out + x