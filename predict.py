import torch
import torch.nn as nn
import torch.nn.functional as F

# instantiate the model
model = torch.load("")

# make predictions on new data
input = torch.randn(1,3,256,256)
output = model(input)
probs = F.softmax(output, dim=1)

print(probs)