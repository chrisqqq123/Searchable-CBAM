'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_module import MVAM, MyChannelAttention, MySpatialAttention


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        # self.cbam = CBAM(planes)
        self.ca = MyChannelAttention(planes)
        self.sa = MySpatialAttention()


    def forward(self, x, alpha):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out_list = []
        out_list.append(out)
        cs_out = self.ca(out) * out
        cs_out = self.sa(cs_out) * cs_out
        out_list.append(cs_out)
        sc_out = self.sa(out) * out
        sc_out = self.ca(sc_out) * sc_out
        out_list.append(sc_out)
        c_out = self.ca(out) * out
        out_list.append(c_out)
        s_out = self.sa(out) *out
        out_list.append(s_out)
        output = 0
        for i in range(len(alpha)):
            if alpha[i]>1e-7:
                output += out_list[i] * alpha[i]
        # out = out * alpha[0] +  cs_out * alpha[1] + sc_out * alpha[2] + alpha[3] * self.ca(out) * out + alpha[4] * self.sa(out) * out 

        
        output += self.shortcut(residual)
        output = F.relu(output)
        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()

        # self.alpha = alpha

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        # self.cbam = CBAM(self.expansion*planes)
        self.ca = MyChannelAttention(self.expansion*planes)
        self.sa = MySpatialAttention()


    def forward(self, x, alpha):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out_list = []
        out_list.append(out)
        cs_out = self.ca(out) * out
        cs_out = self.sa(cs_out) * cs_out
        out_list.append(cs_out)
        sc_out = self.sa(out) * out
        sc_out = self.ca(sc_out) * sc_out
        out_list.append(sc_out)
        c_out = self.ca(out) * out
        out_list.append(c_out)
        s_out = self.sa(out) *out
        out_list.append(s_out)
        output = 0
        for i in range(len(alpha)):
            if alpha[i]>1e-7:
                output += out_list[i] * alpha[i]
        # out = out * alpha[0] +  cs_out * alpha[1] + sc_out * alpha[2] + alpha[3] * self.ca(out) * out + alpha[4] * self.sa(out) * out 

        
        output += self.shortcut(residual)
        output = F.relu(output)
        return output


class ResNetMVAM_search_structure_2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNetMVAM_search_structure_2, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self._arch_names = []
        self._arch_parameters = []
        arch_name, arch_param = self._build_arch_parameters(num_blocks)
        self._arch_names.append(arch_name)
        self._arch_parameters.append(arch_param)
        # print(self._arch_names,self._arch_parameters)
        # print(getattr(self, self._arch_names[0]["alphas"][0]))

        

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            i += 1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _arch_loss(self, ratio = [1,1,1,1,1], wgt = 1):
        loss = 0
        alphas = []
        alphas.append( F.softmax(getattr(self, self._arch_names[0]["alphas"][0]), dim=-1) )
        alphas.append( F.softmax(getattr(self, self._arch_names[0]["alphas"][1]), dim=-1) )
        alphas.append( F.softmax(getattr(self, self._arch_names[0]["alphas"][2]), dim=-1) )
        alphas.append( F.softmax(getattr(self, self._arch_names[0]["alphas"][3]), dim=-1) )
        for i in range(len(alphas)):
            for j in range(len(alphas[i])):
                for k in range(len(alphas[i][j])):
                    loss += ratio[k]*(alphas[i][j][k]**1.1)
                    # print(ratio[k],alphas[i][j][k],betas[i][j][k],'444444444444444444444444')
        return loss*wgt/sum(self.num_blocks)

    def forward(self, x):
        alphas0 = F.softmax(getattr(self, self._arch_names[0]["alphas"][0]), dim=-1)
        alphas1 = F.softmax(getattr(self, self._arch_names[0]["alphas"][1]), dim=-1)
        alphas2 = F.softmax(getattr(self, self._arch_names[0]["alphas"][2]), dim=-1)
        alphas3 = F.softmax(getattr(self, self._arch_names[0]["alphas"][3]), dim=-1)

        out = F.relu(self.bn1(self.conv1(x)))
        i = 0
        for layer in self.layer1:
            out = layer(out, alphas0[i])
            i += 1
        i = 0
        for layer in self.layer2:
            out = layer(out, alphas1[i])
            i += 1
        i = 0
        for layer in self.layer3:
            out = layer(out, alphas2[i])
            i += 1
        i = 0
        for layer in self.layer4:
            out = layer(out, alphas3[i])
            i += 1

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _build_arch_parameters(self, num_blocks):
        # define names
        alphas = [ "alpha_"+str(scale) for scale in [0, 1, 2, 3] ]
        i = 0
        for num in num_blocks:
            setattr(self, alphas[i], nn.Parameter(torch.autograd.Variable(torch.zeros(num, 5).cuda(), requires_grad=True)))
            i += 1
        # betas = [ "beta_"+str(scale) for scale in [0, 1, 2, 3] ]
        # i = 0
        # for num in num_blocks:
        #     setattr(self, betas[i], nn.Parameter(torch.autograd.Variable(torch.zeros(num, 3).cuda(), requires_grad=True)))
        #     i += 1
        return {"alphas": alphas}, [getattr(self, name) for name in alphas] 

def ResNetMVAMSS18_2():
    return ResNetMVAM_search_structure_2(BasicBlock, [2,2,2,2])

def ResNetMVAMSS34_2():
    return ResNetMVAM_search_structure_2(BasicBlock, [3,4,6,3])

def ResNetMVAMSS50_2():
    return ResNetMVAM_search_structure_2(Bottleneck, [3,4,6,3])

def ResNetMVAMSS101_2():
    return ResNetMVAM_search_structure_2(Bottleneck, [3,4,23,3])

def ResNetMVAMSS152_2():
    return ResNetMVAM_search_structure_2(Bottleneck, [3,8,36,3])