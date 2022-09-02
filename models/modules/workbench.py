'''
################################################
# Models being tested and developed            #
################################################
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.building_blocks import ResBlock, ResDCB, ResDCB_SN, ResDCB_ob, ResUCB, ResUCB_SN
from models.modules.xia_2020_learning import TransformerXia


class ResNetGenerator(nn.Module):
    '''
    This structure is inspired in the Self-attention GAN
    in https://github.com/brain-research/self-attention-gan.
    ResUNet inspired from https://arxiv.org/abs/1711.10684.
    '''
    def __init__(self, args):
        super(ResNetGenerator, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Input shape
        self.input_shape = np.asarray(self.args['input_shape'])
        # number of features to start with
        self.num_feat = args['initial_filters']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_space_dim']
        self.num_classes = self.args['num_classes']
        self.n_channels = self.args['n_channels']
        self.use_tanh = self.args['use_tanh']
        self.normalization = self.args['gen_params']['norm']

        if self.args['gen_params']['activation'] == 'lrelu':
            self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        elif self.args['gen_params']['activation'] == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['gen_params']['activation']))

        # Encoding path
        self.enc1 = ResBlock({ # 256x256
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': 1, 'OF': self.num_feat})
        self.enc2 = ResDCB({ # 128*128
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = ResDCB({ # 64*64
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = ResDCB({ # 32*32
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = ResDCB({ # 16*16
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Decoding path
        self.dec4 = ResUCB({ # 16*16
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat*4})
        self.dec3 = ResUCB({ # 32*32
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*2})
        self.dec2 = ResUCB({ # 64*64
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat})
        self.dec1 = ResUCB({ # 128*128
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.dec0 = ResUCB({ # 256*256
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.dec0 = nn.Sequential(
            nn.BatchNorm2d(self.num_feat),
            self.nonlinearity,
            nn.Conv2d(self.num_feat, 1, self.kernel_size, stride=1, padding=1, bias=False),
        )

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat*8,
            'latent_dim': self.latent_dim,
            'image_size': 16,
            'encoding': self.args['gen_params']['encoding'],
            'num_classes': self.num_classes}
        )

        if self.use_tanh:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x, z):
        '''Call function.'''
        # Encoding path
        enc1 = x
        enc2 = self.enc1(enc1)
        # print('\t enc2', enc2.size())
        enc3 = self.enc2(enc2)
        # print('\t enc3', enc3.size())
        enc4 = self.enc3(enc3)
        # print('\t enc4', enc4.size())
        enc5 = self.enc4(enc4)
        # print('\t enc5', enc5.size())
        enc6 = self.enc5(enc5)
        # print('\t enc6', enc6.size())

        # Transformer
        dec5 = self.trans(enc6, z)

        # Decoding path
        # print('\t dec5', dec5.size(), enc5.size())
        dec4 = self.dec4(dec5, enc5)
        # print('\t dec4', dec4.size(), enc4.size())
        dec3 = self.dec3(dec4, enc4)
        # print('\t dec3', dec3.size(), enc3.size())
        dec2 = self.dec2(dec3, enc3)
        # print('\t dec2', dec2.size(), enc2.size())
        dec1 = self.dec1(dec2, enc2)
        # print('\t dec1', dec1.size())

        _map = self.dec0(dec1)

        # Add mapping to input image
        output = _map + x
        output = self.activation(output)

        return output, _map



class ResNetDiscriminator(nn.Module):
    '''
    Residual Discriminator inspired by SAGAN.
    '''
    def __init__(self, args):
        super(ResNetDiscriminator, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Input shape
        self.input_shape = np.asarray(self.args['input_shape'])
        # number of features to start with
        self.num_feat = args['initial_filters']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_space_dim']
        self.num_classes = self.args['num_classes']
        self.normalization = self.args['discr_params']['norm']

        if self.args['discr_params']['activation'] == 'lrelu':
            self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        elif self.args['discr_params']['activation'] == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['discr_params']['activation']))

        # Encoding path
        self.enc1 = ResBlock({ # 256x256
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': 1, 'OF': self.num_feat})
        self.enc2 = ResDCB({ # 128*128
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = ResDCB({ # 64*64
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = ResDCB({ # 32*32
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = ResDCB({ # 16*16
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Divide image size by 2 and multiply feature number by 2
        # Output has dim (BS, 256, 16, 16)
        self.encoder = nn.Sequential(
            self.enc1, self.enc2, self.enc3, self.enc4, self.enc5)

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 16,
            'encoding': self.args['discr_params']['encoding'],
            'num_classes': self.num_classes}
        )

        # Judge
        self.judge = nn.Sequential(
            # nn.Dropout(p=0.5),
            # (BS, 256+32, 8, 8)
            nn.Conv2d(self.num_feat*(8+1), self.num_feat*8,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.Conv2d(self.num_feat*8, self.num_feat*8,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.Conv2d(self.num_feat*8, self.num_feat*8,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # (BS, 256, 8, 8)
            nn.Conv2d(self.num_feat*8, 1,
                      self.kernel_size, stride=1, padding=1, bias=False),
            # self.nonlinearity,
            # (BS, 1, 16, 16)
            nn.AvgPool2d(16) # Average pooling per feature to do global average pooling
                             # One value in the kernel per pixel. This image is 8x8 here.
            # (BS, 1, 1, 1)
        )

    def forward(self, x, z):
        # Encoding path
        enc = self.encoder(x)
        # enc = x
        # for cnt,layer in enumerate(self.encoder):
        #     enc = layer(enc)
        #     print('\n-> (dec) enc{}'.format(cnt+1), torch.max(enc), torch.min(enc))
        #     print('\t name', enc.size(), layer)

        # Transformer
        aux = self.trans(enc, z)
        # print(' -> (dec) aux', torch.max(aux), torch.min(aux))

        # Decoding path
        dec = torch.cat((aux, enc), dim=1)
        # print(' -> (dec) dec', torch.max(dec), torch.min(dec))

        output = self.judge(dec)
        # print(' -> (dec) output', torch.max(output), torch.min(output))
        # output = dec
        # for cnt, layer in enumerate(self.judge):
        #     output = layer(output)
        #     print(' -> (dec) judge{}'.format(cnt+1), torch.max(output), torch.min(output))
        #     print('\t name', output.size(), layer)
        #     if cnt > 5:
        #         for name, param in layer.named_parameters():
        #             print(' \t-> layer params', name, torch.max(param), torch.min(param))

        return output
