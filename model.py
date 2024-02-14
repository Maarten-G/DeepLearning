import torch
import torch.nn as nn

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
model.classifier[1] = nn.Linear(model.last_channel, 10)