import matplotlib as mpl#NOQA
mpl.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from clize import run

from skimage.io import imsave
from sklearn.decomposition import PCA

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from machinedesign.viz import grid_of_images_default

from model import Gen
from model import Discr

from data import load_dataset


from scipy.spatial.distance import cdist
from lapjv import lapjv

def grid_embedding(h):
    assert int(np.sqrt(h.shape[0])) ** 2 == h.shape[0], 'Nb of examples must be a square number'
    size = np.sqrt(h.shape[0])
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))).reshape(-1, 2)
    cost_matrix = cdist(grid, h, "sqeuclidean").astype('float32')
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    _, rows, cols = lapjv(cost_matrix)
    return rows



def save_weights(m, folder='out', prefix=''):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if np.sqrt(w.size(1)) == int(w.size(1)):
            s = int(np.sqrst(w.size(1)))
            w = w.view(w.size(0), 1, s, s)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)
    elif isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        if w.size(1) == 1:
            w = w.view(w.size(0) * w.size(1), w.size(2), w.size(3))
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)
        elif w.size(1) == 3:
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('{}/{}_feat_{}.png'.format(folder, prefix, w.size(0)), gr)


def train(*, folder='out', dataset='celeba', resume=False, wasserstein=True, batch_size=64, nz=100, coef_pred=0., nb_factors=0, nb_blocks=None, nb_convs_per_block=1):
    lr = 0.0002
    nb_epochs = 3000
    dataset = load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    nc = x0.size(0)
    w = x0.size(1)
    h = x0.size(2)
    _save_weights = partial(save_weights, folder=folder, prefix='gan')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    act = 'tanh'
    if resume:
        gen = torch.load('{}/gen.th'.format(folder))
        discr = torch.load('{}/discr.th'.format(folder))
    else:
        if nb_blocks:
            nb_blocks = int(nb_blocks)
        gen = Gen(nz=nz, nc=nc, act=act, w=w, nb_blocks=nb_blocks, nb_convs_per_block=nb_convs_per_block)
        gen.nb_factors = nb_factors
        discr = Discr(nc=nc, act='', w=w, no=nb_factors + 1, nb_blocks=nb_blocks)
    
    if wasserstein:
        gen_opt = optim.RMSprop(gen.parameters(), lr=lr)
        discr_opt = optim.RMSprop(discr.parameters(), lr=lr)
    else:
        gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
        discr_opt = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))

    input = torch.FloatTensor(batch_size, nc, w, h)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    label = torch.FloatTensor(batch_size)

    if wasserstein:
        real_label = 1
        fake_label = -1
        criterion = lambda output, label:(output*label).mean()
    else:
        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()

    gen = gen.cuda()
    discr =  discr.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()

    giter = 0
    diter = 0

    dreal_list = []
    dfake_list = []
    pred_error_list = []

    nb_blocks = nz // gen.nz_per_block
    nb_factors_per_block = nb_factors // nb_blocks

    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            if wasserstein:
                # clamp parameters to a cube
                for p in discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # Update discriminator
            discr.zero_grad()
            batch_size = X.size(0)
            X = X.cuda()
            input.resize_as_(X).copy_(X)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            output = discr(inputv)
            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errD_real = criterion(labelpred, labelv)
            errD_real.backward()
            D_x = labelpred.data.mean()
            dreal_list.append(D_x)
            noise.resize_(batch_size, nz, 1, 1).uniform_(-1, 1)
            noisev = Variable(noise)
            fake = gen(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = discr(fake.detach())
            pred_error = 0.
            for i, j in zip(range(0, nz, gen.nz_per_block), range(0, nb_factors, nb_factors_per_block)):
                pred_error += ((output[:, 1 + j:1 + j + nb_factors_per_block] - noisev[:, i:i+nb_factors_per_block, 0, 0])**2).mean()
            pred_error_list.append(pred_error.data[0])

            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errD_fake = criterion(labelpred, labelv) + coef_pred * pred_error
            errD_fake.backward()
            D_G_z1 = labelpred.data.mean()
            dfake_list.append(D_G_z1)
            discr_opt.step()
            diter += 1
            
            # Update generator
            gen.zero_grad()
            fake = gen(noisev)
            labelv = Variable(label.fill_(real_label))
            output = discr(fake)
            pred_error = 0.
            for i, j in zip(range(0, nz, gen.nz_per_block), range(0, nb_factors, nb_factors_per_block)):
                pred_error += ((output[:, 1 + j:1 + j + nb_factors_per_block] - noisev[:, i:i+nb_factors_per_block, 0, 0])**2).mean()

            labelpred = output[:, 0:1] if wasserstein else nn.Sigmoid()(output[:, 0:1])
            errG = criterion(labelpred, labelv) + coef_pred * pred_error
            errG.backward()

            gen_opt.step()
            print('{}/{} dreal : {:.6f} dfake : {:.6f} pred_error : {:.6f}'.format(epoch, nb_epochs, D_x, D_G_z1, pred_error.data[0]))

            if giter % 100 == 0:
                x = 0.5 * (X + 1) if act == 'tanh' else X
                f = 0.5 * (fake.data + 1) if act == 'tanh' else fake.data
                vutils.save_image(x, '{}/real_samples.png'.format(folder), normalize=True)
                vutils.save_image(f, '{}/fake_samples_epoch_{:03d}.png'.format(folder, epoch), normalize=True)
                torch.save(gen, '{}/gen.th'.format(folder))
                torch.save(discr, '{}/discr.th'.format(folder))
                gen.apply(_save_weights)
               
                fig = plt.figure()
                plt.plot(dreal_list, label='real')
                plt.plot(dfake_list, label='fake')
                plt.legend()
                plt.savefig('{}/discr.png'.format(folder))
                plt.close(fig)

                fig = plt.figure()
                plt.plot(pred_error_list)
                plt.savefig('{}/pred_error.png'.format(folder))
                plt.close(fig)
                generate(folder=folder)
            giter += 1


def generate(*, folder='mnist'):
    gen = torch.load('{}/gen.th'.format(folder), map_location=lambda storage, loc: storage)
    nz = gen.nz
    nb_blocks = nz // gen.nz_per_block
    nb_factors_per_block = gen.nb_factors // nb_blocks
    nb = 20 * 20
    for i in range(0, nz, gen.nz_per_block):
        Z = np.random.uniform(-1, 1, size=(1, nz)) * np.ones((nb, 1))
        Z[:, i:i + nb_factors_per_block] = np.random.uniform(-1, 1, size=(nb, nb_factors_per_block))
        Z = Z.astype(np.float32)
        p = PCA(n_components=2).fit_transform(Z)
        p = grid_embedding(p)

        Z = torch.from_numpy(Z)
        Z = Z.view(Z.size(0), Z.size(1), 1, 1)
        Z = Variable(Z)
        x = gen(Z)
        x = x.data.numpy()
        x = x[p]
        im = grid_of_images_default(x, normalize=True)
        imsave('{}/latent{}.png'.format(folder, i // gen.nz_per_block), im)

if __name__ == '__main__':
    run([train, generate])
