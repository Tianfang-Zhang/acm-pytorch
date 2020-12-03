import sys
import torch
import torch.nn as nn

input = torch.rand(8, 3, 20, 20)
m = nn.AdaptiveAvgPool2d((1, 1))
output = m(input)

print(output.shape)