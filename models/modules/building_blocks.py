''''
Basic blocks for models.
'''
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


class CategoricalConditionalBatchNorm(nn.Module):
    """
    Similar to batch norm, but with per-category weight and bias.
    """
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class BasicBlock(nn.Module):
    '''
    Basic block for handling the parameters (BasicBlock).
    '''
    def __init__(self, args):
        super(BasicBlock, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Input size
        self.H, self.W = self.args['input_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']
        # Non-linear function to apply (ReLU or LeakyReLU)
        self.nonlinearity = self.args['nonlinearity']
        # Normalization layer
        self.normalization = self.args['normalization']
        if self.normalization == 'batchnorm':
            self.normlayer = lambda x1,x2,x3: nn.BatchNorm2d(x1)
            # self.normlayer = lambda x: nn.BatchNorm2d(x)
        elif self.normalization == 'layernorm':
            self.normlayer = lambda x1,x2,x3: nn.LayerNorm([x1,x2,x3])
            # self.normlayer = lambda x: nn.LayerNorm([x, self.H, self.W])
        else:
            self.normlayer = lambda x1,x2,x3: nn.Identity()


class ConvBlock(BasicBlock):
    '''Convolutional block. A module is composed of two.'''
    def __init__(self, args) -> None:
        super().__init__(args)

        self.block = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            self.normlayer(self.OF, self.H, self.W),
            self.nonlinearity,
        )

    def forward(self, x):
        return self.block(x)


class DeConvBlock(BasicBlock):
    '''Transpose convolutional block.'''
    def __init__(self, args) -> None:
        super().__init__(args)

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                self.IF, self.OF, self.kernel_size, stride=2, padding=1, output_padding=1, bias=False),
            self.normlayer(self.OF, self.H, self.W),
            self.nonlinearity,
        )

    def forward(self, x):
        return self.block(x)


class DCB_LN2(BasicBlock):
    '''
    *Corrected version
    Downsampling Convolutional Block with Layer Normalization (DCB_LN2).
    '''
    def __init__(self, args):
        super().__init__(args)

        self.model = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            nn.LayerNorm([self.OF,self.H,self.W]),
            self.nonlinearity,
            nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.LayerNorm([self.OF,self.H,self.W]),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        y = self.model(x)

        return y


class DCB(BasicBlock):
    '''
    Downsampling Convolutional Block (DCB).
    '''
    def __init__(self, args):
        super().__init__(args)

        args['IF'] = self.IF
        args['OF'] = self.OF
        block1 = ConvBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ConvBlock(args)

        self.model = nn.Sequential(
            block1,
            block2,
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        y = self.model(x)

        return y


class DCB_LN_Old(nn.Module):
    '''
    Downsampling Convolutional Block with Layer Normalization (DCB_LN_Old).
    '''
    def __init__(self, args):
        super(DCB_LN_Old, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Input size
        self.H, self.W = self.args['input_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']
        # Non-linear function to apply (ReLU or LeakyReLU)
        self.nonlinearity = self.args['nonlinearity']

        self.model = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            nn.LayerNorm([self.OF,self.H,self.W]),
            self.nonlinearity,
            nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        y = self.model(x)

        return y


class DCB_Old(nn.Module):
    '''
    Downsampling Convolutional Block with Batch Normalization (DCB_Old).
    '''
    def __init__(self, args):
        super(DCB_Old, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']
        # Non-linear function to apply (ReLU or LeakyReLU)
        self.nonlinearity = self.args['nonlinearity']

        self.model = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        y = self.model(x)

        return y


class DCB_SN(DCB):
    '''
    Downsampling Convolutional Block with Spectral Normalization (DCB_SN).
    '''
    def __init__(self, args):
        super().__init__(args)

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            self.nonlinearity,
            spectral_norm(nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=(2, 2))
        )


class ResBlock(BasicBlock):
    '''Residual block. A module is composed of two.'''
    def __init__(self, args) -> None:
        super().__init__(args)

        self.block = nn.Sequential(
            self.normlayer(self.IF, self.H, self.W),
            self.nonlinearity,
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class ResDCB(DCB):
    '''
    Residual Downsampling Convolutional Block (ResDCB).
    '''
    def __init__(self, args):
        super().__init__(args)

        args['IF'] = self.IF
        args['OF'] = self.OF
        block1 = ResBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ResBlock(args)

        self.model = nn.Sequential(
            block1,
            block2,
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        self.res = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, (1,1), stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        x0 = x
        x  = self.model(x)
        x0 = self.res(x0)

        return x + x0


class ResDCB_ob(ResDCB):
    '''
    Residual Downsampling Convolutional Block
    optimized block, i.e. first block (ResDCB_ob).
    '''
    def __init__(self, args):
        super().__init__(args)

        # args['IF'] = self.IF
        # args['OF'] = self.OF
        # block1 = ResBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ResBlock(args)

        self.model = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            # block1,
            block2,
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        self.res = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(self.IF, self.OF, (1,1), stride=1, padding=0, bias=False),
        )


class ResDCB_SN(ResDCB):
    '''
    Residual Downsampling Convolutional Block
    with Spectral Normalization (ResDCB_SN).
    '''
    def __init__(self, args):
        super().__init__(args)

        self.model = nn.Sequential(
            self.nonlinearity,
            spectral_norm(nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            self.nonlinearity,
            spectral_norm(nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        self.res = nn.Sequential(
            spectral_norm(nn.Conv2d(self.IF, self.OF, (1,1), stride=1, padding=0, bias=False), eps=1e-6),
            nn.AvgPool2d(kernel_size=(2, 2))
        )


class UCB(BasicBlock):
    '''
    Upsampling Convolutional Block (UCB).
    '''
    def __init__(self, args):
        super().__init__(args)

        args['IF'] = 2*self.OF
        args['OF'] = self.OF
        block1 = ConvBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ConvBlock(args)

        self.step1 = nn.ConvTranspose2d(
            self.IF, self.OF, self.kernel_size, stride=2, padding=1, output_padding=1, bias=False)
        # TODO: Inverted nonlinearity and normlayer to have the corrected version
        self.step2 = nn.Sequential(
            block1,
            block2,
        )

        # self.step2 = nn.Sequential(block1,block2)

        # args['IF'] = self.IF
        # args['OF'] = self.OF
        # self.step1 = DeConvBlock(args)

    def forward(self, x1, x2):
        x1 = self.step1(x1)
        x = torch.cat((x1, x2), dim=1)
        y = self.step2(x)

        return y


class UCB_Old(nn.Module):
    '''
    Upsampling Convolutional Block (UCB_Old).
    '''
    def __init__(self, args):
        super(UCB_Old, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']

        self.step1 = nn.ConvTranspose2d(
            self.IF, self.OF, self.kernel_size, stride=2, padding=1, output_padding=1, bias=False)
        self.step2 = nn.Sequential(
            nn.Conv2d(2*self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.OF),
            nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.OF),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.step1(x1)
        x = torch.cat((x1, x2), dim=1)
        y = self.step2(x)

        return y


class UCB_SN(UCB):
    '''
    Upsampling Convolutional Block with Spectral Normalization (UCB_SN).
    '''
    def __init__(self, args):
        super().__init__(args)

        self.step1 = nn.ConvTranspose2d(
            self.IF, self.OF, self.kernel_size, stride=2, padding=1, output_padding=1, bias=False)
        self.step2 = nn.Sequential(
            spectral_norm(nn.Conv2d(2*self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            nn.ReLU(inplace=True)
        )


class ResUCB(UCB):
    '''
    Residual Upsampling Convolutional Block (ResUCB).
    '''
    def __init__(self, args):
        super().__init__(args)

        args['IF'] = 2*self.OF
        args['OF'] = self.OF
        block1 = ResBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ResBlock(args)

        self.step1 = nn.ConvTranspose2d(
            self.IF, self.OF, self.kernel_size, stride=2, padding=1,
            output_padding=1, bias=False)
        self.step2 = nn.Sequential(
            block1,
            block2,
        )

        self.res = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.IF, self.OF, (1,1), stride=1, padding=0, bias=False)
        )

    def forward(self, x1, x2):
        x = self.step1(x1)
        x = torch.cat((x, x2), dim=1)
        x = self.step2(x)

        x0 = self.res(x1)

        return x + x0


class ResUCB_SN(ResUCB):
    '''
    Residual Upsampling Convolutional Block
    with Spectral Normalization (ResUCB_SN).
    '''
    def __init__(self, args):
        super().__init__(args)

        self.step2 = nn.Sequential(
            self.nonlinearity,
            spectral_norm(nn.Conv2d(2*self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
            self.nonlinearity,
            spectral_norm(nn.Conv2d(self.OF, self.OF, self.kernel_size, stride=1, padding=1, bias=False), eps=1e-6),
        )

        self.res = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(self.IF, self.OF, (1,1), stride=1, padding=0, bias=False), eps=1e-6),
        )
