'''
GAN module with pytorch lightning logs.

Reference at
https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py
'''
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.utils as vutils

import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import Callback

# from models.regressor import Regressor
# from models.blueprints import vgg
from models.modules.xia_2020_learning import \
    DiscriminatorXiaOld, GeneratorXiaOld, \
    GeneratorXiaSN, DiscriminatorXiaSN,  \
    GeneratorXia, DiscriminatorXia, \
    GeneratorXia2, GeneratorXiaAttention, \
    GeneratorXiaReduced, DiscriminatorXiaReduced
from models.modules.workbench import ResNetGenerator, ResNetDiscriminator
from models.blueprints.unet_diffusion import UNetModel, EncoderUNetModel



def l1_regularization(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def l1_regularization_loss(y_true, y_pred, age_gap, age_range=60):
    # epsilon = torch.exp(-age_gap/age_range)
    epsilon = 1
    l1_loss = epsilon * nn.L1Loss()(y_pred, y_true)

    return torch.mean(l1_loss)

def dice_coef(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred, dim=1)+0.1
    union = torch.sum(y_true, dim=1)+torch.sum(y_pred, dim=1)+0.1
    dice_coef = torch.mean(2*intersection/union, dim=0)
    return dice_coef

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true,y_pred)

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)


class InitCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        print("Sending tensors to GPU when model is initialized!")
        pl_module.reg_weight = pl_module.reg_weight.to(pl_module.device)
        pl_module.cyc_weight = pl_module.cyc_weight.to(pl_module.device)
        pl_module.gp_weight  = pl_module.gp_weight.to(pl_module.device)


def init(args, settings):
    'Initialize GAN model.'
    model = GAN(**settings)
    callback = InitCallback()
    # callback = None

    return model, callback


class GAN(pl.LightningModule):
    """
    WGAN-GP implementation.
    """

    def __init__(
        self,
        num_classes: int = 1,
        **kwargs
    ):
        """
        Args:
            learning_rate: the learning rate
            beta1: first momentum of Adam optimizer
        """
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.kwargs['num_classes'] = num_classes
        self.reg_weight = torch.Tensor(
            [self.hparams.regularization_weight])
        self.cyc_weight = torch.Tensor(
            [self.hparams.cycle_cons_weight])
        self.gp_weight  = torch.Tensor(
            [self.hparams.gradient_penalty_weight])

        # networks
        if self.hparams.reduced:
            self.generator = GeneratorXiaReduced(kwargs)
            self.discriminator = DiscriminatorXiaReduced(kwargs)
        else:
            print('> Generator module is {}'.format(
                self.kwargs['gen_params']['model']))
            gen_module = globals()[self.kwargs['gen_params']['model']]
            self.generator = gen_module(kwargs)
            # self.generator.convert_to_fp16()

            print('> Discriminator module is {}'.format(
                self.kwargs['discr_params']['model']))
            discr_module = globals()[self.kwargs['discr_params']['model']]
            self.discriminator = discr_module(kwargs)
            # self.discriminator.convert_to_fp16()

        # Loss criteria
        self.criterionCycle = torch.nn.L1Loss(reduction='mean')
        self.criterionIdt = torch.nn.L1Loss(reduction='mean')
        # Weight clipping
        self.weight_cliping_limit = 0.01
        # Logging parameter
        self.log_images = False

    def forward(self, *args):
        "Generates an image given input arguments"
        return self.generator(*args)

    def generator_loss(self, batch_idx, x, y, x_lbl, y_lbl):
        '''
        Params:
            batch_idx: batch that is being processed
            x: input image
            y: image corresponding to target class
            x_lbl: input class
            y_lbl: target class
        '''
        ###############################
        # (2) Update generator D: maximize log(D(G(z)))
        ###############################
        y_real = -torch.ones(x.size(0), 1, device=self.device)
        # (2.1) generate images
        generated_imgs, _map = self(x, y_lbl-x_lbl)
        # Generated images with target class
        d_output = self.discriminator(generated_imgs, y_lbl)

        # ground truth result (ie: all real)
        g_loss = wasserstein_loss(d_output, y_real)
        self.log('w_loss', g_loss, on_epoch=True, prog_bar=False)

        # (2.2) and go back again
        reconstr_imgs, _ = self(generated_imgs, x_lbl-y_lbl)

        # Proportion of modification (Map loss with raw mapping)
        raw_map_loss = torch.mean(torch.abs(_map))
        # How far is _map from 0 (no modification) i.e. mean abs deviation from 0
        self.log('train_raw_map_loss', raw_map_loss, on_epoch=True, prog_bar=False)
        # Map loss through the radial prior
        map_loss = torch.mean(torch.abs(_map))
        self.log('map_loss', map_loss, on_epoch=True, prog_bar=False)

        # Cycle-consistency loss
        cyc_loss = self.criterionCycle(reconstr_imgs, x)
        self.log('cyc_loss', cyc_loss, on_epoch=True, prog_bar=False)
        cyc_loss = cyc_loss * self.cyc_weight

        # Regularization loss
        #   To ensure that resulting images are not very different from input images
        reg_loss = self.criterionIdt(generated_imgs, x)
        self.log('reg_loss', reg_loss, on_epoch=True, prog_bar=False)
        reg_loss = reg_loss * self.reg_weight

        # Generator loss
        # g_loss = g_loss + reg_loss + cyc_loss
        g_loss = g_loss + reg_loss + cyc_loss + map_loss
        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)

        if self.log_images and self.kwargs['data_type'] != '3d':
            g_ims = generated_imgs.clip(0, 1)
            _map = (_map - _map.median())/2 + 0.5
            _map = _map.clip(0, 1)
            aux = torch.stack(
                (x, _map, g_ims, y), dim=0).reshape(
                    4*x.size()[0], x.size()[1], x.size()[2], x.size()[3])
            grid = vutils.make_grid(aux, nrow=x.size()[0])
            self.logger.experiment[0].add_image(
                'example_images_epoch_{}'.format(self.current_epoch), grid, 0)
            self.log_images = False

        return g_loss

    def discriminator_loss(self, x, y, x_lbl, y_lbl):
        '''
        Params:
            x: input image
            y: image corresponding to target class
            x_lbl: input class
            y_lbl: target class
        '''
        ###############################
        # (1) Update discriminator D: maximize log(D(x)) + log(1-D(G(z)))
        #     in Wasserstein GAN: - E(D(y)) + E(D(G(x)))
        #                             real       fake
        ###############################
        # (1.1) train discriminator on real
        # calculate real score
        y_real = -torch.ones(x.size(0), 1, device=self.device)
        d_output = self.discriminator(y, y_lbl)
        d_real_loss = wasserstein_loss(d_output, y_real)

        # (1.2) train discriminator on fake
        x_fake, _ = self(x, y_lbl-x_lbl)
        # x_fake, _ = self(x, [y_lbl-x_lbl, x_lbl])
        # calculate fake score
        y_fake = torch.ones(x.size(0), 1, device=self.device)
        d_output = self.discriminator(x_fake, y_lbl)
        d_fake_loss = wasserstein_loss(d_output, y_fake)

        # (1.3) Gradient penalty (WGAN-GP)
        gradient_penalty = self.calc_gradient_penalty(y, x_fake, x_lbl, y_lbl)
        # gradient_penalty = self.calc_dragan_gradient_penalty(y, x_fake, x_lbl, y_lbl)

        # (1.4) add another fake example with real image and wrong label
        # y_fake = torch.ones(x.size(0), 1, device=self.device)
        # d_output = self.discriminator(x, y_lbl)
        # d_fake_label_loss = wasserstein_loss(d_output, y_fake)

        # (1.5) Compute gradient penalty again for fake label sample
        # in this case the two images are the same. The sampling in GP will give x.
        # gradient_penalty_fake_lb = self.calc_gradient_penalty(x, x, x_lbl, y_lbl, a_or_b='b')

        # gradient backprop & optimize ONLY d's parameters
        self.log('d_real_loss', d_real_loss, on_epoch=True, prog_bar=False)
        self.log('d_fake_loss', d_fake_loss, on_epoch=True, prog_bar=False)
        # self.log('d_fake_label_loss', d_fake_label_loss, on_epoch=True, prog_bar=False)
        # d_loss = d_real_loss + d_fake_loss + d_fake_label_loss
        d_loss = d_real_loss + d_fake_loss
        self.log('d_loss_no_gp', d_loss, on_epoch=True, prog_bar=False)
        # d_loss = d_loss + self.gp_weight * (gradient_penalty + gradient_penalty_fake_lb)
        d_loss = d_loss + self.gp_weight * gradient_penalty
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        # use the discriminator loss for checkpointing
        self.log('train_loss', -d_loss, on_epoch=True, prog_bar=False)

        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        or_im, or_lbl, *_ = batch['A']
        tg_im, tg_lbl, *_ = batch['B']

        # train generator
        result = None
        ncritic = self.hparams.ncritic
        if self.current_epoch < self.hparams.warming_epochs:
            # Update generator less frequently
            ncritic *= 10

        if optimizer_idx == 0:
            # Run training only when necessary.
            if (batch_idx + 1) % ncritic != 0:
                # No update. Sleep loop for Generator.
                return result
            elif batch_idx + 1 == ncritic and self.current_epoch % 10 == 0:
                # Criteria for logging images
                self.log_images = True

            # Train generator.
            result = self.generator_step(batch_idx, or_im, tg_im, or_lbl, tg_lbl)

        # Train discriminator.
        if optimizer_idx == 1:
            result = self.discriminator_step(or_im, tg_im, or_lbl, tg_lbl)

        return result

    def generator_step(self, batch_idx, x, y, x_z, y_z):
        g_loss = self.generator_loss(batch_idx, x, y, x_z, y_z)

        return g_loss

    def discriminator_step(self, x, y, x_lbl, y_lbl):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x, y, x_lbl, y_lbl)

        return d_loss

    def validation_step(self, batch, batch_idx):
        or_im, or_lbl, *_ = batch['A']
        tg_im, tg_lbl, *_ = batch['B']

        x_fake, _map = self(or_im, tg_lbl-or_lbl)

        # Map loss with raw mapping
        raw_map_loss = torch.mean(torch.abs(_map))
        # How far is _map from 0 (no modification) i.e. mean abs deviation from 0
        self.log('val_raw_map_loss', raw_map_loss, on_epoch=True, prog_bar=False)

        pass

    def test_step(self, batch, batch_idx):
        or_im, or_lbl, *im_data = batch['A']

        results_dir = self.hparams.results_folder
        os.makedirs(results_dir, exist_ok=True)
        samples_dir = os.path.join(results_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        numpy_dir = os.path.join(results_dir, 'numpy')
        os.makedirs(numpy_dir, exist_ok=True)

        # Generate sample of target images
        older_ims = []
        younger_ims = []
        for i in range(30):
            # assuming batch size of 1
            # Getting older
            tg_lbl = im_data[2] + i
            x_fake, _map = self(or_im, tg_lbl-or_lbl)
            x_fake = x_fake.clip(0, 1)
            _map = (_map - _map.median())/2 + 0.5
            _map = _map.clip(0, 1)
            older_ims.append(
                (x_fake.cpu().numpy().squeeze(), _map.cpu().numpy().squeeze()))
            # Getting younger
            tg_lbl = im_data[2] - i
            x_fake, _map = self(or_im, tg_lbl-or_lbl)
            x_fake = x_fake.clip(0, 1)
            _map = (_map - _map.median())/2 + 0.5
            _map = _map.clip(0, 1)
            younger_ims.append(
                (x_fake.cpu().numpy().squeeze(), _map.cpu().numpy().squeeze()))

        # Filename
        if batch_idx < 20:
            filename = '{}_sample_{:.0f}_sex-{:.0f}_age-{:.0f}_bmi-{:.1f}.png'
            filename = os.path.join(samples_dir, filename)
            # Older subject
            fig = plt.figure(figsize=(20,15))
            or_im = (or_im - or_im.min())/(or_im.max() - or_im.min())
            row1 = np.hstack([or_im.cpu().numpy().squeeze()]*6)
            # Plot only steps of 5 years
            row2 = np.hstack([m for _,m in older_ims[::5]]) # Maps
            row3 = np.hstack([i for i,_ in older_ims[::5]]) # Imgs
            grid = np.vstack([row1,row2,row3])
            plt.imshow(grid, cmap='gray')
            fig.savefig(filename.format(
                'old', im_data[0].item(), im_data[1].item(),
                im_data[2].item(), im_data[3].item()), dpi=300)
            plt.close()

            # Younger subject
            fig = plt.figure(figsize=(20,15))
            row1 = np.hstack([or_im.cpu().numpy().squeeze()]*6)
            # Plot only steps of 5 years
            row2 = np.hstack([m for _,m in younger_ims[:26][::-1][::5]]) # Maps
            row3 = np.hstack([i for i,_ in younger_ims[:26][::-1][::5]]) # Imgs
            grid = np.vstack([row1,row2,row3])
            plt.imshow(grid, cmap='gray')
            fig.savefig(filename.format(
                'yng', im_data[0].item(), im_data[1].item(),
                im_data[2].item(), im_data[3].item()), dpi=300)
            plt.close()

        # ---- Save numpy arrays ----
        filename = 'sample_{:.0f}_sex-{:.0f}_age-{:.0f}_bmi-{:.1f}_{}_step_{}'
        filename = os.path.join(numpy_dir, filename)
        # Getting older
        for idx, (im, map) in enumerate(older_ims):
            np.save(filename.format(
                im_data[0].item(), im_data[1].item(),
                im_data[2].item(), im_data[3].item(), 'image', idx), im)
            np.save(filename.format(
                im_data[0].item(), im_data[1].item(),
                im_data[2].item(), im_data[3].item(), 'map', idx), map)
        # Rejuvenating
        for idx, (im, map) in enumerate(younger_ims):
            if idx == 0:
                continue
            np.save(filename.format(
                im_data[0].item(), im_data[1].item(),
                im_data[2].item(), im_data[3].item(), 'image', -idx), im)
            np.save(filename.format(
                im_data[0].item(), im_data[1].item(),
                im_data[2].item(), im_data[3].item(), 'map', -idx), map)

    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(self, epoch, batch_idx, optimizer,
        optimizer_idx, optimizer_closure, on_tpu=False, 
        using_native_amp=False, using_lbfgs=False):

        ncritic = self.hparams.ncritic
        if self.current_epoch < self.hparams.warming_epochs:
            # Update generator less frequently
            ncritic *= 10
        # update discriminator every step
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)

        # update generator every ncritic steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % ncritic == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # optional: call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()

    def configure_optimizers(self):
        try:
            # TTUR: Two time-scale update rule
            lrate_d = self.hparams.learning_rate_disc
            lrate_g = self.hparams.learning_rate_gen
        except Exception:
            lrate   = self.hparams.learning_rate
            lrate_d = lrate
            lrate_g = lrate
        beta1   = self.hparams.beta1
        beta2   = self.hparams.beta2
        weightd = self.hparams.weight_decay

        if self.kwargs['optG'] == 'adam':
            opt_g = optim.Adam(
                self.generator.parameters(), lr=lrate_g, betas=(beta1, beta2),
                weight_decay=weightd)
        elif self.kwargs['optG'] == 'radam':
            # Rectified ADAM. Prevents large bias at the start of training.
            # Should work similarly to using a warmup.
            opt_g = torch.optim.RAdam( # https://arxiv.org/pdf/1908.03265.pdf
                self.generator.parameters(), lr=lrate_g, betas=(beta1, beta2),
                weight_decay=weightd)
        elif self.kwargs['optG'] == 'adamw':
            opt_g = optim.AdamW(
                self.generator.parameters(), lr=lrate_g, betas=(beta1, beta2),
                weight_decay=weightd)
        else:
            opt_g = optim.RMSprop(
                self.generator.parameters(), lr=lrate_g, weight_decay=weightd)

        if self.kwargs['optD'] == 'adam':
            opt_d = optim.Adam(
                self.discriminator.parameters(), lr=lrate_d, betas=(beta1, beta2),
                weight_decay=weightd)
        elif self.kwargs['optD'] == 'radam':
            # Rectified ADAM. Prevents large bias at the start of training.
            # Should work similarly to using a warmup.
            opt_d = torch.optim.RAdam( # https://arxiv.org/pdf/1908.03265.pdf
                self.discriminator.parameters(), lr=lrate_g, betas=(beta1, beta2),
                weight_decay=weightd)
        elif self.kwargs['optD'] == 'adamw':
            opt_d = optim.AdamW(
                self.discriminator.parameters(), lr=lrate_d, betas=(beta1, beta2),
                weight_decay=weightd)
        else:
            opt_d = optim.RMSprop(
                self.discriminator.parameters(), lr=lrate_d, weight_decay=weightd)

        # G is idx 0, D is idx 1
        return opt_g, opt_d

    def calc_gradient_penalty(self, real_data, fake_data, real_lbl, fake_lbl, a_or_b='a'):
        'Compute grandient penalty for WGAN-GP (Arxiv:1704.00028)'
        batch_size = real_data.size(0)
        if self.kwargs['data_type'] == '2d':
            epsilon = torch.rand(batch_size, 1, 1, 1)
        else:
            epsilon = torch.rand(batch_size, 1, 1, 1, 1)
        epsilon = epsilon.expand_as(real_data).to(self.device)

        # Gradient w.r.t. 1st variable
        # Averaged images
        if a_or_b == 'a':
            interpolation_im = epsilon * real_data + (1 - epsilon) * fake_data
            interpolation_im = autograd.Variable(
                interpolation_im, requires_grad=True).to(self.device)

            # TODO: Check the validity of using the target class index together
            # with the interpolated image for the gradient penalty loss.
            interpolation_logits = self.discriminator(interpolation_im, fake_lbl)
            grad_outputs = torch.ones(interpolation_logits.size(), device=self.device)
            gradients_a = autograd.grad(outputs=interpolation_logits,
                                        inputs=interpolation_im,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Gradient w.r.t. 2nd variable
        # TODO: Test this gradient with a non-integer encoding.
        # Averaged labels
        elif a_or_b == 'b':
            e2 = torch.rand(batch_size)
            e2 = e2.expand_as(real_lbl).to(self.device)
            int_lbl = e2 * real_lbl + (1 - e2) * fake_lbl
            # We need to cast it to long for the different encoding steps
            int_lbl = autograd.Variable(int_lbl, requires_grad=True).to(self.device)

            interpolation_logits = self.discriminator(real_data, int_lbl)
            grad_outputs = torch.ones(interpolation_logits.size(), device=self.device)
            gradients_b = autograd.grad(outputs=interpolation_logits,
                                        inputs=int_lbl,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        if a_or_b == 'a':
            gradients = gradients_a.view(batch_size, -1)
        elif a_or_b == 'b':
            gradients = gradients_b.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        self.log('gradient_pty_{}'.format(a_or_b), gradient_penalty, on_epoch=True, prog_bar=False)

        return gradient_penalty

    def calc_dragan_gradient_penalty(self, real_data, fake_data, real_lbl, fake_lbl, a_or_b='a'):
        '''
        DRAGAN (Deep Regret Analytic Generative Adversarial Networks)
        Code from https://github.com/kodalinaveen3/DRAGAN and
        https://github.com/jfsantos/dragan-pytorch
        '''
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size,  1, 1, 1).expand(real_data.size()).to(self.device)
        noisy_data = real_data.data + 0.5 * real_data.data.std() * torch.rand(real_data.size()).to(self.device)
        x_hat = autograd.Variable(
            alpha * real_data.data + (1 - alpha) * noisy_data, requires_grad=True).to(self.device)
        pred_hat = self.discriminator(x_hat, fake_lbl)
        grad_outputs = torch.ones(pred_hat.size(), device=self.device)
        gradients = autograd.grad(
            outputs=pred_hat, inputs=x_hat,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
