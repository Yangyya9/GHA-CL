import math
from torch import nn
from torch.nn.functional import normalize
from torch.nn import init
from losses import InstanceContrastiveModule, ClassContrastiveModule
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class GHA_CL_MVC(nn.Module):
    def __init__(self, args, view, input_size, high_feature_dim, low_feature_dim, device, contrastive_ins_enable, contrastive_cls_enable):
        super(GHA_CL_MVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], high_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], high_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.Specific_view = nn.Sequential(
            nn.Linear(high_feature_dim, low_feature_dim),
        )

        self.Common_view = nn.Sequential(
            nn.Linear(high_feature_dim * view, low_feature_dim),
        )

        self.view = view
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=high_feature_dim * view, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)

        self.contrastive_ins_enable = contrastive_ins_enable
        self.contrastive_cls_enable = contrastive_cls_enable

        if self.contrastive_ins_enable: #是否启用实例级对比学习
            self.instance_loss = InstanceContrastiveModule(args)
        if self.contrastive_cls_enable:
            self.cls_loss = ClassContrastiveModule(args, device)


    def contrastive_loss(self,commonz,hs):
        if self.contrastive_ins_enable:
            loss_ins = self.instance_loss.get_loss(commonz, hs)
        else:
            loss_ins = 0
        if self.contrastive_cls_enable:
            loss_cls = self.cls_loss.get_loss(commonz, hs)
        else:
            loss_cls = 0
        # 得到对比损失函数: λ1*实例级损失+λ2*类级损失
        contrastive_loss = 0.5 * loss_ins + 0.5 * loss_cls

        return contrastive_loss, loss_ins, loss_cls

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return xrs, zs, hs

    def GHA_CL(self, xs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        commonz_cat = torch.cat(zs, 1)
        commonz, S = self.TransformerEncoderLayer(commonz_cat)
        commonz = normalize(self.Common_view(commonz), dim=1)
        commonz_cat = normalize(self.Common_view(commonz_cat), dim=1)
        return commonz, commonz_cat, S

    def First_z(self, zs):
        firstz=torch.cat(zs,1)
        firstz=normalize(self.Common_view(firstz), dim=1)
        return firstz


