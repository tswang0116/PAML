import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function

class hash(Function):
    @staticmethod
    def forward(ctx, U):
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N/2), D]), -torch.ones([N - int(N/2), D]))).cuda()    
        B = torch.zeros(U.shape).cuda().scatter_(0, index, B_creat)
        ctx.save_for_backward(U, B) 
        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B)/(B.numel())

        grad = g + gamma*add_g

        return grad

def hash_layer(input):
    return hash.apply(input)

class CNN(nn.Module):
    def __init__(self, encode_length):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False
        torch.manual_seed(0)           
        self.fc_encode = nn.Linear(4096, encode_length)

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        h = self.fc_encode(x)
        b = hash_layer(h)
        return x, h, b