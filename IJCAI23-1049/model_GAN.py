import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from max_sv import max_singular_value

class SNLinear(nn.modules.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)   # sigma 最大奇异值
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)

class Discriminator(nn.Module):
    def __init__(self, input_n=25):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(

            SNLinear(input_n, 128),
            nn.ReLU(),
            SNLinear(128, 64),
            nn.ReLU(),
            SNLinear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch_size * nodes * frame
        x = x.reshape(x.shape[0], -1) # 16 * 25
        x = self.dis(x)
        x = x.reshape(x.shape[0])
        return x