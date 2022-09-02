'''
################################################
# Models in Xia et al. 1920.02620              #
# Learning to synthesize the ageing brain      #
################################################
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.building_blocks import DCB, DCB_LN2, DCB_SN, UCB, UCB_SN, DCB_LN_Old, DCB_Old, UCB_Old


class RandomFourierFeatures:
    def __init__(self, embedding_size: int = 5, features_size: int  = 1):
        scale = 4
        self.embedding_size = embedding_size
        self.features_size = features_size
        torch.manual_seed(1234)
        self.W = torch.randn((features_size, embedding_size)) * scale

    def get_random_fourier_embedding(self, x):
        assert x.shape[-1] == self.W.shape[0]
        x_proj = (x @ self.W) * 2 * np.pi # [bs,features,embedding_size]
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_fourier

    def __call__(self, x):
        assert x.shape[-1] == self.W.shape[0]
        dvc = x.get_device()
        self.W = self.W.to(dvc)
        x_proj = torch.matmul(x, self.W) * 2 * np.pi # [bs,features,embedding_size]
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_fourier


class OrdinalEncoding:
    def __init__(self, embedding_size: int = 45):
        self.enc_half_size = embedding_size
        weights = torch.ones((2*self.enc_half_size, 2*self.enc_half_size))
        for i in range(2*self.enc_half_size):
            weights[i, i:].zero_()
        self.embedding_matrix = weights
        self.thr = nn.Threshold(0, 0)

    def __call__(self, x):
        'Transform 1-dim. vector into ordinal encoding vector.'
        vec = x.long()
        gap = self.enc_half_size
        # Apply thresholding (i.e. max(0,x))
        enc_up = self.thr(vec) + gap
        enc_do = -self.thr(-vec) + gap

        dvc = vec.get_device()
        self.embedding_matrix = self.embedding_matrix.to(dvc)
        # Embed in vectors of ones until given value (e.g. 40 gives a 80-dim. vector with ones for half of it)
        enc_up = F.embedding(enc_up, self.embedding_matrix, padding_idx=0)
        # Embed in vectors of -ones until given value (e.g. 40 gives a 80-dim. vector with ones for half of it)
        # then add one and get the vectors with ones from a given value onwards.
        enc_do = 1 + F.embedding(enc_do, -self.embedding_matrix, padding_idx=0)
        # Obtain intersection (i.e. ones from the middle up to the age diff).
        # E.g. [+2] is [0 0 1 1 0 0 0 0] for gap = 4
        # E.g. [-2] is [0 0 0 0 1 1 0 0] for gap = 4
        enc = enc_up * enc_do

        # Size of final tensor is (BS, 1, 2*gap).
        # (BS, 2*gap) after squeeze.
        return enc.squeeze(dim=1)


class FourierFeatures:
    def __init__(self, embedding_matrix):
        self.W = embedding_matrix

    def get_random_fourier_embedding(self, x):
        assert x.shape[-1] == self.W.shape[0]
        x_proj = (x @ self.W) * 2 * np.pi # [bs,features,embedding_size]
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_fourier

    def __call__(self, x):
        assert x.shape[-1] == self.W.shape[0]
        dvc = x.get_device()
        self.W = self.W.to(dvc)
        x_proj = torch.matmul(x, self.W) * 2 * np.pi # [bs,features,embedding_size]
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_fourier


class TransformerXia(nn.Module):
    '''
    Combination of fully connected layers with input vectors.
    '''
    def __init__(self, args):
        super(TransformerXia, self).__init__()

        self.args = args
        # Half size of age encoding space
        self.enc_half_size = 45
        self.enc_vec_size = 2*self.enc_half_size
        # Embedding matrix
        weights = torch.ones((2*self.enc_half_size, 2*self.enc_half_size))
        for i in range(2*self.enc_half_size):
            weights[i, i:].zero_()
        self.embedding_matrix = weights
        self.thr = nn.Threshold(0, 0)

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_dim']
        self.num_classes = self.args['num_classes']
        # Size of smallest image
        self.image_size = self.args['image_size']

        if self.args['encoding'] == 'positive':
            # Ordinal encoding (OE1) - only positive
            self.encoding_operation = self._vec_to_enc_pve
        elif self.args['encoding'] == 'both':
            # Ordinal encoding (OE2) - positive and negative
            self.encoding_operation = self._vec_to_enc
        elif self.args['encoding'] == 'fourier':
            self.encoding_operation = RandomFourierFeatures(self.enc_half_size)
        else:
            # No encoding
            self.encoding_operation = nn.Identity()
            self.enc_vec_size = 1

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.OF),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # 15 * 15 * 32 = 7200
            nn.Linear(self.image_size*self.image_size*self.OF, self.latent_dim),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.latent_dim)
        )

        # Transform a vector through a NN
        self.trans = nn.Sequential(
            # nn.Linear(self.latent_dim + self.num_classes, self.latent_dim),
            # +ve and -ve ordinal encoding
            # Ordinal encoding (OE1)
            nn.Linear(self.latent_dim + self.enc_vec_size, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.latent_dim)
        )

        # Decoder
        self.dec = nn.Sequential(
            # One variable version
            nn.Linear(self.latent_dim, self.image_size*self.image_size*self.OF),
            # Two variable version
            # nn.Linear(self.latent_dim + self.num_classes, self.image_size*self.image_size*self.OF),
            # nn.Linear(self.latent_dim + 2*self.enc_half_size, self.image_size*self.image_size*self.OF),
            nn.ReLU(inplace=True)
        )

    def _vec_to_enc(self, vec):
        'Transform 1-dim. vector into ordinal encoding vector.'
        # This step does remove the grad_fn attribute from vec
        # probably because it becomes an integer (non-differentiable) vector.
        vec = vec.long()
        gap = self.enc_half_size
        # Apply thresholding (i.e. max(0,x))
        enc_up = self.thr(vec) + gap
        enc_do = -self.thr(-vec) + gap

        dvc = vec.get_device()
        if dvc > -1:
            self.embedding_matrix = self.embedding_matrix.to(dvc)
        # Embed in vectors of ones until given value (e.g. 40 gives a 80-dim. vector with ones for half of it)
        enc_up = F.embedding(enc_up, self.embedding_matrix, padding_idx=0)
        # Embed in vectors of -ones until given value (e.g. 40 gives a 80-dim. vector with ones for half of it)
        # then add one and get the vectors with ones from a given value onwards.
        enc_do = 1 + F.embedding(enc_do, -self.embedding_matrix, padding_idx=0)
        # Obtain intersection (i.e. ones from the middle up to the age diff).
        # E.g. [+2] is [0 0 1 1 0 0 0 0] for gap = 4
        # E.g. [-2] is [0 0 0 0 1 1 0 0] for gap = 4
        enc = enc_up * enc_do

        # Size of final tensor is (BS, 1, 2*gap).
        # (BS, 2*gap) after squeeze.
        return enc.squeeze(dim=1)

    def _vec_to_enc_pve(self, vec):
        'Transform 1-dim. positive vector into ordinal encoding vector.'
        # This step does remove the grad_fn attribute from vec
        vec = vec.long()
        dvc = vec.get_device()
        if dvc > -1:
            self.embedding_matrix = self.embedding_matrix.to(dvc)
        # Embed in vectors of ones until given value
        # (e.g. 40 gives a 80-dim. vector with ones for half of it)
        enc = F.embedding(vec, self.embedding_matrix, padding_idx=0)

        # Size of final tensor is (BS, 1, 2*self.enc_half_size).
        # (BS, 2*self.enc_half_size) after squeeze.
        return enc.squeeze(dim=1)

    def forward(self, x, vec):
        # TODO: Pass vec only once or add another variable
        vec = vec.reshape(vec.size(0), 1)
        vec = self.encoding_operation(vec)
        z1 = self.enc(x)
        z1cat = torch.cat((z1, vec), dim=1)
        z2 = self.trans(z1cat)
        # One variable version
        z2cat = z2
        # Two variable version
        # z2cat = torch.cat((z2, vec), dim=1)
        z = self.dec(z2cat)

        out = torch.reshape(z, (x.size(0), self.OF, x.size(2), x.size(3)))

        return out


class GeneratorXia(nn.Module):
    '''
    This structure follows the paper design in 1920.02620 and not exactly the
    github code at https://github.com/xiat0616/BrainAgeing. There are differences.
    '''
    def __init__(self, args):
        super(GeneratorXia, self).__init__()

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
        elif self.args['gen_params']['activation'] == 'gelu':
            self.nonlinearity = nn.GELU()
        elif self.args['gen_params']['activation'] == 'silu':
            self.nonlinearity = nn.SiLU(inplace=True)
        elif self.args['gen_params']['activation'] == 'mish':
            self.nonlinearity = nn.Mish(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['gen_params']['activation']))

        # Encoding path
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
        )
        self.enc2 = DCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 16,
            'encoding': self.args['gen_params']['encoding'],
            # 'encoding': 'both',
            'num_classes': self.num_classes}
        )

        # Decoding path
        self.dec4 = UCB({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*(8+1), 'OF': self.num_feat*4})
        self.dec3 = UCB({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*2})
        self.dec2 = UCB({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat})
        self.dec1 = UCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat, self.input_shape[0], self.input_shape[1]]),
            self.nonlinearity,
            # nn.ConvTranspose2d(
            #     self.num_feat, self.num_feat, self.kernel_size, stride=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            # self.nonlinearity # Should be chosen with the experiment
            # nn.Conv2d(
            #     self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat)
            nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False)
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
        enc3 = self.enc2(enc2)
        enc4 = self.enc3(enc3)
        enc5 = self.enc4(enc4)
        enc6 = self.enc5(enc5)

        # Transformer
        aux = self.trans(enc6, z)

        # Decoding path
        dec5 = torch.cat((aux, enc6), dim=1)
        dec4 = self.dec4(dec5, enc5)
        dec3 = self.dec3(dec4, enc4)
        dec2 = self.dec2(dec3, enc3)
        dec1 = self.dec1(dec2, enc2)

        _map = self.dec0(dec1)

        # Add mapping to input image
        output = _map + x
        output = self.activation(output)

        return output, _map


class GeneratorXiaOld(nn.Module):
    '''
    This structure follows the paper design in 1920.02620 and not exactly the
    github code at https://github.com/xiat0616/BrainAgeing. There are differences.
    '''
    def __init__(self, args):
        super(GeneratorXiaOld, self).__init__()

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
        elif self.args['gen_params']['activation'] == 'gelu':
            self.nonlinearity = nn.GELU()
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['gen_params']['activation']))

        # Encoding path
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
        )
        self.enc2 = DCB_LN_Old({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB_LN_Old({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB_LN_Old({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB_LN_Old({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 8,
            'encoding': self.args['gen_params']['encoding'],
            # 'encoding': 'both',
            'num_classes': self.num_classes}
        )

        # Decoding path
        self.dec4 = UCB_Old({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*(8+1), 'OF': self.num_feat*4})
        self.dec3 = UCB_Old({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*2})
        self.dec2 = UCB_Old({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat})
        self.dec1 = UCB_Old({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
            self.nonlinearity,
            # nn.ConvTranspose2d(
            #     self.num_feat, self.num_feat, self.kernel_size, stride=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            # self.nonlinearity # Should be chosen with the experiment
            # nn.Conv2d(
            #     self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat)
            nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False)
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
        enc3 = self.enc2(enc2)
        enc4 = self.enc3(enc3)
        enc5 = self.enc4(enc4)
        enc6 = self.enc5(enc5)

        # Transformer
        aux = self.trans(enc6, z)

        # Decoding path
        dec5 = torch.cat((aux, enc6), dim=1)
        dec4 = self.dec4(dec5, enc5)
        dec3 = self.dec3(dec4, enc4)
        dec2 = self.dec2(dec3, enc3)
        dec1 = self.dec1(dec2, enc2)

        _map = self.dec0(dec1)

        # Add mapping to input image
        output = _map + x
        output = self.activation(output)

        return output, _map


class GeneratorXia2(GeneratorXia):
    '''
    *Corrected version.
    This structure follows the paper design in 1920.02620 and not exactly the
    github code at https://github.com/xiat0616/BrainAgeing. There are differences.
    '''
    def __init__(self, args):
        super().__init__(args)

        # Encoding path
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
            # nn.BatchNorm2d(self.num_feat),
            self.nonlinearity,
        )
        self.enc2 = DCB_LN2({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB_LN2({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB_LN2({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB_LN2({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 16,
            'encoding': self.args['gen_params']['encoding'],
            # 'encoding': 'both',
            'num_classes': self.num_classes}
        )

        # Decoding path
        self.dec4 = UCB({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*(8+1), 'OF': self.num_feat*4})
        self.dec3 = UCB({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*2})
        self.dec2 = UCB({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat})
        self.dec1 = UCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat, self.input_shape[0], self.input_shape[1]]),
            self.nonlinearity,
            nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False),
            nn.LayerNorm([self.n_channels, self.input_shape[0], self.input_shape[1]]),
            # nn.BatchNorm2d(self.n_channels),
        )


class GeneratorXiaSN(nn.Module):
    '''
    This structure follows the paper design in 1920.02620 and not exactly the
    github code at https://github.com/xiat0616/BrainAgeing. There are differences.
    '''
    def __init__(self, args):
        super(GeneratorXiaSN, self).__init__()

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
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
        )
        self.enc2 = DCB_SN({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB_SN({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB_SN({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB_SN({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 16,
            'encoding': self.args['gen_params']['encoding'],
            # 'encoding': 'both',
            'num_classes': self.num_classes}
        )

        # Decoding path
        self.dec4 = UCB_SN({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity, 
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*(8+1), 'OF': self.num_feat*4})
        self.dec3 = UCB_SN({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity, 
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*2})
        self.dec2 = UCB_SN({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity, 
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat})
        self.dec1 = UCB_SN({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity, 
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
            self.nonlinearity,
            # nn.ConvTranspose2d(
            #     self.num_feat, self.num_feat, self.kernel_size, stride=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            # self.nonlinearity # Should be chosen with the experiment
            # nn.Conv2d(
            #     self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat)
            nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False)
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
        enc3 = self.enc2(enc2)
        enc4 = self.enc3(enc3)
        enc5 = self.enc4(enc4)
        enc6 = self.enc5(enc5)

        # Transformer
        aux = self.trans(enc6, z)

        # Decoding path
        dec5 = torch.cat((aux, enc6), dim=1)
        dec4 = self.dec4(dec5, enc5)
        dec3 = self.dec3(dec4, enc4)
        dec2 = self.dec2(dec3, enc3)
        dec1 = self.dec1(dec2, enc2)

        _map = self.dec0(dec1)

        # Add mapping to input image
        output = _map + x
        output = self.activation(output)

        return output, _map


class GeneratorXiaAttention(GeneratorXia):
    '''
    Attention map as in Liu et al. A3GAN arxiv:1911.06531.
    *They use kernel sizes of 7x7 in first and last layers of generator.
    '''
    def __init__(self, args):
        super().__init__(args)

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
            self.nonlinearity
        )

        self.outMi = nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size,
                stride=1, padding=1, bias=False)
        self.outMa = nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size,
                stride=1, padding=1, bias=False)

    def forward(self, x, z):
        '''Call function.'''
        # Encoding path
        enc1 = x
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)
        enc4 = self.enc3(enc3)
        enc5 = self.enc4(enc4)
        enc6 = self.enc5(enc5)

        # Transformer
        aux = self.trans(enc6, z)

        # Decoding path
        dec5 = torch.cat((aux, enc6), dim=1)
        dec4 = self.dec4(dec5, enc5)
        dec3 = self.dec3(dec4, enc4)
        dec2 = self.dec2(dec3, enc3)
        dec1 = self.dec1(dec2, enc2)

        dec0 = self.dec0(dec1)
        int_map = self.outMi(dec0)
        att_map = self.outMa(dec0)

        # Add mapping to input image with attention
        output = att_map * int_map + (1-att_map) * x
        output = self.activation(output)

        return output, (int_map, att_map)


class DiscriminatorXia(nn.Module):
    '''A simpler version than GeneratorXia with fewer layers.'''
    def __init__(self, args):
        super(DiscriminatorXia, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
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
        elif self.args['discr_params']['activation'] == 'gelu':
            self.nonlinearity = nn.GELU()
        elif self.args['discr_params']['activation'] == 'silu':
            self.nonlinearity = nn.SiLU(inplace=True)
        elif self.args['discr_params']['activation'] == 'mish':
            self.nonlinearity = nn.Mish(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['discr_params']['activation']))

        # Encoding path. Shapes - (batch size, filters, rows, cols)
        # TODO: Sustitute batch normalization by layer normalization, otherwise
        # the gradient penalty is not penalizing the gradients for each input
        # but for the whole batch and the loss is no longer effective.
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.BatchNorm2d(self.num_feat),
        )
        self.enc2 = DCB({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB({'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB({'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB({'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
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
            # 'encoding': 'positive',
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
            # (BS, 1, 8, 8)
            nn.AvgPool2d(16) # Average pooling per feature to do global average pooling
                             # One value in the kernel per pixel. This image is 8x8 here.
            # (BS, 1, 1, 1)
        )

    def forward(self, x, z):
        # Encoding path
        enc = self.encoder(x)
        # print('\n-> (dec) enc', torch.max(enc), torch.min(enc))
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


class DiscriminatorXiaOld(nn.Module):
    '''A simpler version than GeneratorXia with fewer layers.'''
    def __init__(self, args):
        super(DiscriminatorXiaOld, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
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
        elif self.args['discr_params']['activation'] == 'gelu':
            self.nonlinearity = nn.GELU()
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['discr_params']['activation']))

        # Encoding path. Shapes - (batch size, filters, rows, cols)
        # TODO: Sustitute batch normalization by layer normalization, otherwise
        # the gradient penalty is not penalizing the gradients for each input
        # but for the whole batch and the loss is no longer effective.
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.BatchNorm2d(self.num_feat),
        )
        self.enc2 = DCB_Old({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB_Old({'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB_Old({'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB_Old({'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
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
            'image_size': 8,
            'encoding': self.args['discr_params']['encoding'],
            # 'encoding': 'positive',
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
            nn.AvgPool2d(8) # Average pooling per feature to do global average pooling
                             # One value in the kernel per pixel. This image is 8x8 here.
            # (BS, 1, 1, 1)
        )

    def forward(self, x, z):
        # Encoding path
        enc = self.encoder(x)
        # print('\n-> (dec) enc', torch.max(enc), torch.min(enc))
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


class DiscriminatorXiaSN(nn.Module):
    '''A simpler version than GeneratorXia with fewer layers.'''
    def __init__(self, args):
        super(DiscriminatorXiaSN, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
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

        # Encoding path. Shapes - (batch size, filters, rows, cols)
        # TODO: Sustitute batch normalization by layer normalization, otherwise
        # the gradient penalty is not penalizing the gradients for each input
        # but for the whole batch and the loss is no longer effective.
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.BatchNorm2d(self.num_feat),
        )
        self.enc2 = DCB_SN({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB_SN({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB_SN({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB_SN({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
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
            # 'encoding': 'positive',
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
        # print('\n-> (dec) enc', torch.max(enc), torch.min(enc))
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


class GeneratorXiaReduced(nn.Module):
    '''Reduced version of GeneratorXia.'''
    def __init__(self, args):
        super(GeneratorXiaReduced, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
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
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.n_channels, self.num_feat,
                self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.BatchNorm2d(self.num_feat),
        )
        self.enc2 = DCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 128,
            'num_classes': self.num_classes}
        )

        # Decoding path
        self.dec1 = UCB({
            'kernel_size': self.kernel_size, 'IF': self.num_feat*(1+1), 'OF': self.num_feat})

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(self.num_feat),
            self.nonlinearity,
            nn.ConvTranspose2d(
                self.num_feat, self.num_feat, self.kernel_size, stride=1, bias=False),
            # nn.Conv2d(
            #     self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.n_channels),
            # self.nonlinearity # Should be chosen with the experiment
            nn.Conv2d(
                self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_feat),
            nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False)
        )

        if self.use_tanh:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x, idx):
        '''Call function.'''
        # Encoding path
        enc1 = x
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)
        # print('\n-> enc2', enc2.size())
        # print('\n-> enc3', enc3.size())

        # Transformer
        aux = self.trans(enc3, idx)
        # print('\n-> aux', aux.size())

        # Decoding path
        dec2 = torch.cat((aux, enc3), dim=1)
        dec1 = self.dec1(dec2, enc2)
        # print('\n-> dec2', dec2.size())
        # print('\n-> dec1', dec1.size())

        _map = self.dec0(dec1)
        # print('\n-> _map', _map.size())

        # Add mapping to input image
        output = _map + x
        output = self.activation(output)
        # print('\n-> output', output.size())

        return output, _map


class DiscriminatorXiaReduced(nn.Module):
    '''A reduced version of DiscriminatorXia.'''
    def __init__(self, args):
        super(DiscriminatorXiaReduced, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        self.input_shape = np.asarray(self.args['input_shape'])
        # number of features to start with
        self.num_feat = args['initial_filters']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_space_dim']
        self.num_classes = self.args['num_classes']
        self.n_channels = self.args['n_channels']
        self.normalization = self.args['discr_params']['norm']

        if self.args['discr_params']['activation'] == 'lrelu':
            self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        elif self.args['discr_params']['activation'] == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['discr_params']['activation']))

        # Encoding path. Shapes - (batch size, filters, rows, cols)
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.n_channels, self.num_feat,
                self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.BatchNorm2d(self.num_feat),
        )
        self.enc2 = DCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'IF': self.num_feat, 'OF': self.num_feat})

        self.encoder = nn.Sequential(self.enc1, self.enc2)

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': 128,
            'num_classes': self.num_classes}
        )

        # Judge
        self.judge = nn.Sequential(
            # nn.Dropout(p=0.5),
            # (BS, 64, 128, 128)
            nn.Conv2d(self.num_feat*(1+1), self.num_feat,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # (BS, 32, 128, 128)
            nn.AvgPool2d(128) # Average pooling per feature to do global average pooling
                             # One value in the kernel per pixel.
            # (BS, 32, 1, 1)
        )

    def forward(self, x, idx):
        # Encoding path
        enc = self.encoder(x)
        # enc = x
        # for layer in self.encoder:
        #     enc = layer(enc)
        #     print(enc.size(), layer)

        # Transformer
        aux = self.trans(enc, idx)

        # Decoding path
        dec = torch.cat((aux, enc), dim=1)
        # print('\n-> dec', dec.size())

        output = self.judge(dec)
        # enc = dec
        # for cnt,layer in enumerate(self.judge):
        #     enc = layer(enc)
        #     print('\n-> (dec) enc{}'.format(cnt+1), torch.max(enc), torch.min(enc))
        #     print('\t name', enc.size(), layer)
        # output = enc

        # print('\n-> output', output.size())

        return output
