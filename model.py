import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform

class Gen(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, act='tanh', w=64, nb_blocks=None, nb_convs_per_block=1):
        super().__init__()
        self.act = act
        self.nz = nz
        if nb_blocks is None:
            # 32x32   = 5 - 3 = 2 blocks
            # 64x64   = 6 - 3 = 3 blocks
            # 128x128 = 7 - 3 = 4 blocks
            # 256x256 = 8 - 3 = 5 blocks
            # etc.
            nb_blocks = int(np.log(w)/np.log(2)) - 3 
        nf = ngf * 2**(nb_blocks + 1)
        nz_per_block = nz // (nb_blocks + 1)
        self.nz_per_block = nz_per_block

        block = [
            nn.ConvTranspose2d(nz_per_block, nf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
        ]
        self.pre_block = block
        layers = block[:]

        blocks = []
        for _ in range(nb_blocks):
            block = [
                nn.ConvTranspose2d(nf + nz_per_block, nf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ]
            for i in range(nb_convs_per_block - 1):
                block.extend([
                    nn.Conv2d(nf // 2, nf // 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(nf // 2),
                    nn.ReLU(True),
                ])
            blocks.append(block)
            layers.extend(block)
            nf = nf // 2
        self.blocks = blocks
        block = [
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False)
        ]
        self.post_block = block
        layers.extend(block)
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, z):
        i = 0
        x = z[:, i:i + self.nz_per_block]
        i += self.nz_per_block
        for lay in self.pre_block:
            x = lay(x)
        for j, block in enumerate(self.blocks):
            zi = z[:, i:i+self.nz_per_block]
            i += self.nz_per_block
            zi = zi.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat((x, zi), 1)
            for lay in block:
                x = lay(x)
        for lay in self.post_block:
            x = lay(x)
        out = x
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out


class Discr(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', no=1, w=64, nb_blocks=None):
        super().__init__()
        self.act = act
        self.no = no
        if nb_blocks is None:
            nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        layers.append(
            nn.Conv2d(nf, no, 4, 1, 0, bias=False)
        )
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out.view(-1, self.no)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)

