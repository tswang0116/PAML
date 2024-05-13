import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision

# Image subnet
class ImageSubnet(nn.Module):
    def __init__(self, bit):
        super(ImageSubnet, self).__init__()
        self.bit = bit
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential()
        for p in self.parameters():
            p.requires_grad = False
        self.surrogate_model = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, self.bit),
            nn.Tanh())

    def forward(self, feat):
        feat = self.model(feat)
        feat = self.surrogate_model(feat)
        return feat

    def generate_hash_code(self, data):
        num_data = data.size(0)
        feats = torch.zeros(num_data, self.bit).cuda()
        for i in range(num_data):
            feat = self.model(data[i].type(torch.float).unsqueeze(0).cuda())
            feat = self.surrogate_model(feat)
            feats[i, :] = feat
        return torch.sign(feats)


# Text subnet
class TextSubnet(nn.Module):
    def __init__(self, input_dim, bit):
        super(TextSubnet, self).__init__()
        self.bit = bit
        self.surrogate_model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, self.bit),
            nn.Tanh())

    def forward(self, feat):
        feat = self.surrogate_model(feat)
        return feat

    def generate_hash_code(self, data):
        num_data = data.size(0)
        feats = torch.zeros(num_data, self.bit).cuda()
        for i in range(num_data):
            feat = self.surrogate_model(data[i].type(torch.float).unsqueeze(0).cuda())
            feats[i, :] = feat
        return torch.sign(feats)


# ResidualBlock (used in Diffusion model)
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine))
    def forward(self, x):
        return x + self.main(x)


# Diffusion model
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        curr_dim = 64
        image_encoder = [
            nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True)
        ]
        for i in range(2):
            image_encoder += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim * 2
        for i in range(3):
            image_encoder += [ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')]
        self.image_encoder = nn.Sequential(*image_encoder)
        decoder = []
        for i in range(3):
            decoder += [ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')]
        for i in range(2):
            decoder += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2
        self.residual = nn.Sequential(nn.Conv2d(curr_dim + 3, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh())
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, x, noise, time_steps):
        x_noisy = torch.cat((x, noise), dim=1)
        # x_noisy = x + noise
        for _ in range(time_steps):
            encode = self.image_encoder(x_noisy)
            decode = self.decoder(encode)
            decode_x = torch.cat([decode, x], dim=1)
            protected_x = self.residual(decode_x)
        return protected_x


# L2 Normalize (Used in SpectralNorm)
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


# SpectralNorm (Used in spectral_norm)
class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()
        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1),
                          requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)
            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)
        del module._parameters[name]
        setattr(module, name, fn.compute_weight(module))
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


# Spectral_Norm (Used in Discriminator)
def spectral_norm(module):
    SpectralNorm.apply(module)
    return module


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out = self.fc(h)
        return out.squeeze(-1).squeeze(-1)


# Pre-training model
'''
def PretrainingModel2(model_name='resnet50'):
    available_models = torchvision.models.__dict__
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {', '.join(available_models.keys())}")
    model = available_models[model_name](pretrained=True)
    return model
'''
class PretrainingModel(nn.Module):
    def __init__(self, model_name='resnet50'):
        super(PretrainingModel, self).__init__()
        available_models = torchvision.models.__dict__
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not available. Choose from: {', '.join(available_models.keys())}")
        self.base_model = available_models[model_name](pretrained=True)

    def forward(self, x):
        x = self.base_model(x)
        x = F.softmax(x, dim=1)
        return x


# GAN Objective Function
class GANLoss(nn.Module):
    def __init__(self, target_real_label=0.0, target_fake_label=1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss