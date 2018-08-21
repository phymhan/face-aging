import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from util.util import expand2d
import numpy as np

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'instance_affine':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = IdentityMapping
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


# define Generator
def define_G(input_nc, output_nc, nz, ngf, which_model_netG='unet_128', norm='batch', nl='relu',
             dropout=0, init_type='xavier', gpu_ids=[], upsample='bilinear'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, nz, ngf, norm_layer=norm_layer, dropout=dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, nz, ngf, norm_layer=norm_layer, dropout=dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, dropout=dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, dropout=dropout)
    elif which_model_netG == 'unet_128_input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_128_all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256_input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256_all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


# define Discriminator
def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', num_Ds=1, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=n_layers_D, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


# define Identity-Preserving Network
def define_IP(which_model_netIP, input_nc, gpu_ids=[]):
    netIP = None

    if which_model_netIP == 'alexnet':
        netIP = AlexNetFeature(input_nc=input_nc, pooling='None')
    else:
        raise NotImplementedError('Identity-preserving model name [%s] is not recognized' % which_model_netIP)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netIP.to(gpu_ids[0])
        netIP = torch.nn.DataParallel(netIP, gpu_ids)
    return netIP  # do not init weights netIP here, weights will be reloaded anyways


# define Auxiliary Classifier
def define_AC(which_model_netAC, input_nc=3, init_type='normal', num_classes=0, pooling='avg', dropout=0.5, gpu_ids=[]):
    netAC = None

    if which_model_netAC == 'alexnet':
        netAC = AlexNet(input_nc=input_nc, num_classes=num_classes)
    elif which_model_netAC == 'alexnet_lite':
        netAC = AlexNetLite(input_nc=input_nc, num_classes=num_classes, pooling=pooling, dropout=dropout)
    elif 'resnet' in which_model_netAC:
        netAC = ResNet(input_nc=input_nc, num_classes=num_classes, which_model=which_model_netAC)
    else:
        raise NotImplementedError('Auxiliary classifier name [%s] is not recognized' % which_model_netAC)

    return init_net(netAC, init_type, gpu_ids)


# define Auxiliary Regression Network
def define_AR(which_model_netAR, input_nc=3, init_type='kaiming', pooling='max',
              cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2, gpu_ids=[]):
    # Encoder is a Regression Network
    netAR = None

    if which_model_netAR == 'alexnet':
        base = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'resnet' in which_model_netAR:
        base = ResNetFeature(input_nc=input_nc, which_model=which_model_netAR)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_netAR)

    # define Regression Network
    netAR = RegressionNetwork(base, pooling=pooling, cnn_dim=cnn_dim, cnn_pad=cnn_pad, cnn_relu_slope=cnn_relu_slope)

    return init_net(netAR, init_type, gpu_ids)


# define Embedding Encoder
def define_E(which_model_netE, input_nc=3, init_type='kaiming', pooling='max',
             cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2, gpu_ids=[]):
    # Encoder is a Siamese Feature Network
    netE = None

    if which_model_netE == 'alexnet':
        base = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'resnet' in which_model_netE:
        base = ResNetFeature(input_nc=input_nc, which_model=which_model_netE)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_netE)

    # define Siamese Feature Network
    netE = SiameseFeature(base, pooling=pooling, cnn_dim=cnn_dim, cnn_pad=cnn_pad, cnn_relu_slope=cnn_relu_slope)

    return init_net(netE, init_type, gpu_ids)


# define a Siamese Network
def define_S(which_model_netS, input_nc=3, init_type='kaiming', num_classes=3, pooling='max', cnn_dim=[], cnn_pad=1,
             cnn_relu_slope=0.2, fc_dim=[], fc_relu_slope=0.2, dropout=0.5, no_cxn=False, gpu_ids=[]):
    # AC for faceaging_embedding model is a Siamese Network
    netS = None

    if which_model_netS == 'alexnet':
        base = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'resnet' in which_model_netS:
        base = ResNetFeature(input_nc=input_nc, which_model=which_model_netS)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_netS)

    # define Siamese Network
    netS = SiameseNetwork(base, num_classes, pooling=pooling, cnn_dim=cnn_dim, cnn_pad=cnn_pad, cnn_relu_slope=cnn_relu_slope,
                          fc_dim=fc_dim, fc_relu_slope=fc_relu_slope, dropout=dropout, no_cxn=no_cxn)

    return init_net(netS, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

# class GANLoss(nn.Module):
#     def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
#         super(GANLoss, self).__init__()
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#         if use_lsgan:
#             self.loss = nn.MSELoss()
#         else:
#             self.loss = nn.BCELoss()
#
#     def get_target_tensor(self, input, target_is_real):
#         if target_is_real:
#             target_tensor = self.real_label
#         else:
#             target_tensor = self.fake_label
#         return target_tensor.expand_as(input)
#
#     def __call__(self, input, target_is_real):
#         target_tensor = self.get_target_tensor(input, target_is_real)
#         return self.loss(input, target_tensor)


# Allows mix of True (1) and False (0) in a batch
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.Tensor = tensor

    def get_target_tensor(self, input, target_label):
        # target_label is a list indexed along batch
        if not isinstance(target_label, list):
            target_label = [target_label]
        target_label_new = []
        for target_label_item in target_label:
            if isinstance(target_label_item, bool):
                target_label_item = 1 if target_label_item else 0
            target_label_new.append(target_label_item)
        target_tensor = self.Tensor(np.array(target_label_new).reshape(len(target_label_new), 1, 1, 1))
        return target_tensor.expand_as(input)

    def __call__(self, inputs, target_label):
        if not isinstance(inputs, list):
            inputs = [inputs]
        # if input is a list
        loss = 0.0
        all_losses = []
        for input in inputs:
            target_tensor = self.get_target_tensor(input, target_label)
            loss_input = self.loss(input, target_tensor)
            loss += loss_input
            all_losses.append(loss_input)
        return loss


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nz=0, ngf=64, norm_layer=nn.BatchNorm2d, dropout=0, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        input_nc = input_nc + nz  # add
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(input.size(0), z.size(1), input.size(2), input.size(3))
        input_and_z = torch.cat((input, z_img), 1)
        return self.model(input_and_z)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if dropout > 0:
            conv_block += [nn.Dropout(dropout)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nz=0, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, dropout=0):
        super(UnetGenerator, self).__init__()
        input_nc = input_nc + nz  # add
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dropout=dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, z=None):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(input.size(0), z.size(1), input.size(2), input.size(3))
        input_and_z = torch.cat((input, z_img), 1)
        return self.model(input_and_z)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, dropout=0):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if dropout > 0:
                model = down + [submodule] + up + [nn.Dropout(dropout)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


###############################################################################
# Siamese Networks
###############################################################################
class SiameseNetwork(nn.Module):
    def __init__(self, base=None, num_classes=3, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.5, fc_dim=[],
                 fc_relu_slope=0.2, dropout=0.5, no_cxn=False):
        super(SiameseNetwork, self).__init__()
        assert (num_classes == fc_dim[-1])
        self.pooling = pooling
        # base
        self.base = base
        # additional cnns
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim) - 1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        # connection
        cxn_blocks = [
            nn.BatchNorm2d(cnn_dim[-1]),
            nn.LeakyReLU(cnn_relu_slope)
        ]
        self.cxn = None if no_cxn else nn.Sequential(*cxn_blocks)
        # fc layers
        fc_blocks = []
        nf_prev = feature_dim * 2
        for i in range(len(fc_dim) - 1):
            nf = fc_dim[i]
            fc_blocks += [
                nn.Dropout(dropout),
                nn.Conv2d(nf_prev, nf, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(fc_relu_slope)
            ]
            nf_prev = nf
        fc_blocks += [
            nn.Dropout(dropout),
            nn.Conv2d(nf_prev, fc_dim[-1], kernel_size=1, stride=1, padding=0, bias=True)
        ]
        self.fc = nn.Sequential(*fc_blocks)
        self.feature_dim = feature_dim

    def forward_once(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.cxn:
            output = self.cxn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output

    def forward(self, input1, input2):
        feature1 = self.forward_once(input1)
        feature2 = self.forward_once(input2)
        output = torch.cat((feature1, feature2), dim=1)
        output = self.fc(output)
        return feature1, feature2, output

    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn and self.fc won't be initialized
        self.base.load_pretrained(state_dict)


class SiameseFeature(nn.Module):
    def __init__(self, base=None, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2):
        super(SiameseFeature, self).__init__()
        self.pooling = pooling
        self.base = base
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim) - 1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output

    def load_pretrained(self, state_dict):
        # load state dict from a SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        # remove cxn and fc
        for key in list(state_dict.keys()):
            if key.startswith('cxn') or key.startswith('fc'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)


# RegressionNetwork is almost a SiameseFeature Network but with different load_pretrained function
class RegressionNetwork(nn.Module):
    def __init__(self, base=None, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2):
        super(RegressionNetwork, self).__init__()
        self.pooling = pooling
        self.base = base
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim)-1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output

    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn won't be initialized
        self.base.load_pretrained(state_dict)


class AlexNet(nn.Module):
    def __init__(self, input_nc=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier.6'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)


# a lighter alexnet, with fewer params in fc layers
class AlexNetLite(nn.Module):
    def __init__(self, input_nc=3, num_classes=10, pooling='avg', dropout=0.5):
        super(AlexNetLite, self).__init__()
        self.pooling = pooling
        fw = 1 if pooling else 6
        self.features = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * fw * fw, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'avg':
            x = nn.AvgPool2d(x.size(2))(x)
        elif self.pooling == 'max':
            x = nn.MaxPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)


class AlexNetFeature(nn.Module):
    def __init__(self, input_nc=3, pooling='max'):
        super(AlexNetFeature, self).__init__()
        self.pooling = pooling
        sequence = [
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.features = nn.Sequential(*sequence)
        self.feature_dim = 256

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'avg':
            x = nn.AvgPool2d(x.size(2))(x)
        elif self.pooling == 'max':
            x = nn.MaxPool2d(x.size(2))(x)
        return x

    def load_pretrained(self, state_dict):
        # invoked when used as `base' in SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)


class ResNet(nn.Module):
    def __init__(self, input_nc=3, num_classes=0, which_model='resnet18'):
        super(ResNet, self).__init__()
        self.input_nc = input_nc
        model = None
        if which_model == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(False)
            model.fc = nn.Linear(512 * 1, num_classes)
        elif which_model == 'resnet34':
            from torchvision.models import resnet34
            model = resnet34(False)
            model.fc = nn.Linear(512 * 1, num_classes)
        elif which_model == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(False)
            model.fc = nn.Linear(512 * 4, num_classes)
        elif which_model == 'resnet101':
            from torchvision.models import resnet101
            model = resnet101(False)
            model.fc = nn.Linear(512 * 4, num_classes)
        elif which_model == 'resnet152':
            from torchvision.models import resnet152
            model = resnet152(False)
            model.fc = nn.Linear(512 * 4, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('fc'):
                state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


class ResNetFeature(nn.Module):
    def __init__(self, input_nc=3, which_model='resnet18'):
        super(ResNetFeature, self).__init__()
        model = None
        feature_dim = None
        self.input_nc = input_nc
        if which_model == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(False)
            feature_dim = 512 * 1
        elif which_model == 'resnet34':
            from torchvision.models import resnet34
            model = resnet34(False)
            feature_dim = 512 * 1
        elif which_model == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(False)
            feature_dim = 512 * 4
        elif which_model == 'resnet101':
            from torchvision.models import resnet101
            model = resnet101(False)
            feature_dim = 512 * 4
        elif which_model == 'resnet152':
            from torchvision.models import resnet152
            model = resnet152(False)
            feature_dim = 512 * 4
        delattr(model, 'fc')
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


###############################################################################
# Unet from BicycleGAN
###############################################################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, dropout=0,
                 gpu_ids=[], upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(ngf*max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for _ in range(num_downs - 5):
            unet_block = UnetBlock(ngf*max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, dropout=dropout, upsample=upsample)
        unet_block = UnetBlock(ngf*4, ngf * 4, ngf * max_nchn, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf*2, ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc+nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z
        return self.model(x_with_z)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, dropout=0, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)  # downsample is different from upsample
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if dropout > 0:
                model = down + [submodule] + up + [nn.Dropout(dropout)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, dropout=0, gpu_ids=[], upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, dropout=dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, dropout=dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*4, ngf * 4, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*2, ngf * 2, ngf * 4, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, dropout=0, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)  # downsample is different from upsample
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if dropout > 0:
                up += [nn.Dropout(dropout)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            # z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(x.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


###############################################################################
# DRN Generators
###############################################################################
class DRNGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nz=1):
        super(DRNGenerator, self).__init__()
        self.nz = nz
        dropout = 0.8

        self.block0 = nn.Sequential(
            nn.Conv2d(input_nc + nz, 64, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.ReLU(True),
        )

        self.block1_ = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, dilation=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Tanh()
        )

    def forward(self, input, z):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), input.size(2), input.size(3))
        input_and_z = torch.cat([input, z_img], 1)
        x = self.block0(input_and_z)
        x = self.block1_(x) + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        output = self.block4(x)
        return output


###############################################################################
# Multiscale Discriminators
###############################################################################
class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.gpu_ids = gpu_ids
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            for i in range(num_D-1):
                ndf = int(round(ndf/(2**(i+1))))
                layers = self.get_layers(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self, input):
        if self.num_D == 1:
            return self.parallel_forward(self.model, input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D-1:
                down = self.parallel_forward(self.down, down)
        return result


###############################################################################
# Utils
###############################################################################
# Identity mapping
class IdentityMapping(nn.Module):
    def __init__(self, *args):
        super(IdentityMapping, self).__init__()

    def forward(self, x):
        return x


# Channel-wise normalization, similar to torchvision.transforms.Normalize but performed on a 4-D tensor
class Normalize(nn.Module):
    def __init__(self, mean=[], std=[]):
        super(Normalize, self).__init__()
        self.nc = len(mean)
        self.identity_mapping = mean or std  # either one is empty
        if not self.identity_mapping:
            mean_tensor = self.Tensor(np.array(mean).view([1, self.nc, 1, 1]))
            std_tensor = self.Tensor(np.array(std).view([1, self.nc, 1, 1]))
            # mean and std adjusted here, since image is pre-normalized to [-1, 1] instead of [0, 1]
            mean_tensor = 2 * mean_tensor - 1
            std_tensor = 2 * std_tensor
            self.register_buffer('mean', mean_tensor)
            self.register_buffer('std', std_tensor)

    def __call__(self, input):
        if self.identity_mapping:
            return input
        else:
            return (input - self.mean.expand_as(input)) / self.std.expand_as(input)
