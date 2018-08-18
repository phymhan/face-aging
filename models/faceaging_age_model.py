import torch
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from util.util import upsample2d, expand2d, expand2d_as
from .base_model import BaseModel
from . import networks
import random
import numpy as np
from collections import OrderedDict

from util.util import parse_age_label


# TODO: set random seed
class FaceAgingAgeModel(BaseModel):
    def name(self):
        return 'FaceAgingAgeModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        parser.add_argument('--norm_G', type=str, default='batch', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='batch', help='instance normalization or batch normalization')
        parser.add_argument('--embedding_nc', type=int, default=10, help='# of embedding channels')
        parser.add_argument('--which_model_netE', type=str, default='alexnet', help='model type for E loss')
        parser.add_argument('--pooling_E', type=str, default='max', help='which pooling layer in E')
        parser.add_argument('--cnn_dim_E', type=int, nargs='+', default=[64, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--cnn_pad_E', type=int, default=1, help='padding of cnn layers defined by cnn_dim_E')
        parser.add_argument('--cnn_relu_slope_E', type=float, default=0.5, help='slope of LeakyReLU for SiameseNetwork.cnn module')
        parser.add_argument('--fineSize_E', type=int, default=224, help='fineSize for AC')
        parser.add_argument('--pretrained_model_path_E', type=str, default='pretrained_models/alexnet.pth', help='pretrained model path to E net')
        parser.add_argument('--embedding_mean', type=float, nargs='*', default=[0.0], help='means of embedding')
        parser.add_argument('--embedding_std', type=float, nargs='*', default=[100], help='stds of embedding')
        parser.add_argument('--display_aging_visuals', action='store_true', help='display aging visuals if True')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=0.001, help='weight for L1 loss')
            parser.add_argument('--lambda_IP', type=float, default=0.1, help='weight for identity preserving loss')
            parser.add_argument('--lambda_E', type=float, default=1, help='weight for encoder reconstruction loss')
            parser.add_argument('--lambda_A', type=float, default=1, help='weight for cycle consistency loss')
            parser.add_argument('--which_model_netIP', type=str, default='alexnet', help='model type for IP loss')
            parser.add_argument('--fineSize_IP', type=int, default=224, help='fineSize for IP')
            parser.add_argument('--pretrained_model_path_IP', type=str, default='pretrained_models/alexnet.pth', help='pretrained model path to IP net')
            parser.add_argument('--aging_visual_embedding_path', type=str, default='pretrained_models/features.npy', help='pregenerated age embeddings')
            parser.add_argument('--lr_E', type=float, default=0.0002, help='learning rate for E')
            parser.add_argument('--no_trick', action='store_true')
            parser.add_argument('--identity_preserving_criterion', type=str, default='mse', help='which criterion to use for identity preserving loss')
            parser.add_argument('--embedding_reconstruction_criterion', type=str, default='mse', help='which criterion to use for embedding reconstruction loss')
            parser.add_argument('--relabel_D', type=int, nargs='*', default=[0, 1, 0], help='Relabel mapping for Discriminator, 1 for True (label/embedding and image match), 0 for False (don\'t match)')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.input_nc == opt.output_nc)
        assert(opt.embedding_nc == 1)
        self.opt.num_classes = len(opt.age_binranges)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_IP', 'G_L1', 'G_cycle', 'D_real_right', 'D_real_wrong', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'rec_A']
        else:
            self.visual_names = ['real_A']
        # FIXME: disable E/AC for now
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.embedding_nc, opt.ngf,
                                      which_model_netG=opt.which_model_netG,
                                      norm=opt.norm_G, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, upsample=opt.upsample)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # define netD
            self.netD = networks.define_D(opt.embedding_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm_D, use_sigmoid, opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
            # define netIP, which is not saved
            self.netIP = networks.define_IP(opt.which_model_netIP, opt.input_nc, self.gpu_ids)
            if isinstance(self.netIP, torch.nn.DataParallel):
                self.netIP.module.load_pretrained(opt.pretrained_model_path_IP)
            else:
                self.netIP.load_pretrained(opt.pretrained_model_path_IP)

        if self.isTrain:
            # TODO: use num_classes pools
            assert(opt.pool_size == 0)
            self.fake_B_pool = [ImagePool(opt.pool_size) for _ in range(self.opt.num_classes)]
            # define loss functions
            self.criterionGAN = networks.GANLoss2(use_lsgan=not opt.no_lsgan).to(self.device)  # GANLoss2
            # self.criterionGAN = networks.GANLoss(mse_loss=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIP = torch.nn.MSELoss()
            if opt.identity_preserving_criterion.lower() == 'mse':
                self.criterionIP = torch.nn.MSELoss()
            elif opt.identity_preserving_criterion.lower() == 'l1':
                self.criterionIP = torch.nn.L1Loss()
            else:
                raise NotImplementedError('Not Implemented')
            if opt.embedding_reconstruction_criterion.lower() == 'mse':
                self.criterionRec = torch.nn.MSELoss()
            elif opt.embedding_reconstruction_criterion.lower() == 'l1':
                self.criterionRec = torch.nn.L1Loss()
            else:
                raise NotImplementedError('Not Implemented')
            self.criterionCycle = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr_E, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_E)

        self.embedding_mean = opt.embedding_mean
        self.embedding_std = opt.embedding_std
        self.embedding_normalize = lambda x: (x - opt.embedding_mean[0]) / opt.embedding_std[0]

        if opt.display_aging_visuals:
            self.pre_generate_embeddings()

        if self.isTrain:
            self.transform_IP = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), self.use_gpu)

        if self.isTrain:
            self.relabel_D = opt.relabel_D

    def pre_generate_embeddings(self):
        fixed_embeddings = []
        embeddings_npy = np.array(self.opt.age_binranges).reshape(self.opt.num_classes, 1, 1, 1, 1)
        for L in range(embeddings_npy.shape[0]):
            fixed_embeddings.append(self.embedding_normalize(torch.Tensor(embeddings_npy[L]).to(self.device)))
        self.fixed_embeddings = fixed_embeddings

    def set_input(self, input):
        if self.isTrain:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.real_A_IP = upsample2d(self.real_A, self.opt.fineSize_IP)
            # self.real_A_E = upsample2d(self.real_A, self.opt.fineSize_E)
            # self.real_B_E = upsample2d(self.real_B, self.opt.fineSize_E)
            self.age_A = input['A_age'].to(self.device)
            self.age_B = input['B_age'].to(self.device)
            self.label_AB = input['label']
            self.image_paths = input['B_paths']
        else:
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']
        self.current_iter += 1
        self.current_batch_size = int(self.real_A.size(0))

    def forward(self):
        # FIXME: embeddings detached here, fix me if updating E, see BicycleGAN
        self.embedding_A = self.embedding_normalize(self.age_A)
        self.embedding_B = self.embedding_normalize(self.age_B)
        self.fake_B = self.netG(self.real_A, self.embedding_B)
        self.fake_B_IP = upsample2d(self.fake_B, self.opt.fineSize_IP)
        self.rec_A = self.netG(self.fake_B, self.embedding_A)

    def test(self):
        return

    def backward_D(self):
        # Fake image with label_B
        # stop backprop to the generator by detaching fake_B
        # TODO: query from pool
        fake_B = torch.cat((self.fake_B, expand2d(self.embedding_B, self.opt.fineSize)), 1)
        pred_fake = self.netD(fake_B.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_fake = self.criterionGAN(pred_fake, [0])

        # Real_B image with embedding_B
        real_B_embedding_B = torch.cat((self.real_B, expand2d(self.embedding_B, self.opt.fineSize)), 1)
        pred_real = self.netD(real_B_embedding_B)
        # self.loss_D_real_right = self.criterionGAN(pred_real, True)
        self.loss_D_real_right = self.criterionGAN(pred_real, [1])

        if not self.opt.no_trick:
            # Real_B image with embedding_A
            real_B_embedding_A = torch.cat((self.real_B, expand2d(self.embedding_A, self.opt.fineSize)), 1)
            pred_real = self.netD(real_B_embedding_A)
            # self.loss_D_real_wrong = self.criterionGAN(pred_real, False)
            target_label = [self.relabel_D[L] for L in self.label_AB]
            self.loss_D_real_wrong = self.criterionGAN(pred_real, target_label)
            # Combined loss
            self.loss_D = (self.loss_D_fake + (self.loss_D_real_right + self.loss_D_real_wrong) * 0.5) * 0.5
        else:
            self.loss_D_real_wrong = 0.0
            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real_right) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = torch.cat((self.fake_B, expand2d(self.embedding_B, self.opt.fineSize)), 1)
        pred_fake = self.netD(fake_B)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN = self.criterionGAN(pred_fake, [1])

        # L1: fake_B ~= real_A
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        # IP loss
        feature_A = self.netIP(self.transform_IP(self.real_A_IP)).detach()
        feature_A.requires_grad = False
        self.loss_G_IP = self.criterionIP(self.netIP(self.transform_IP(self.fake_B_IP)), feature_A) * self.opt.lambda_IP

        # FIXME: no AC/E for now
        # # Embedding reconstruction loss
        # pred_embedding = self.embedding_normalize(self.netE(self.transform_E(self.fake_B_E)))
        # # print('realA: %.3f, realB: %.3f, fakeB: %.3f' % (self.embedding_A[0,0,0,0], self.embedding_B[0,0,0,0], pred_embedding[0,0,0,0]))  # debug
        # self.loss_z_rec = self.criterionRec(pred_embedding, self.embedding_B) * self.opt.lambda_E

        # Cycle loss
        self.loss_G_cycle = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_IP + self.loss_G_L1  + self.loss_G_cycle  # + self.loss_z_rec

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # TODO: update E

        # update G
        self.set_requires_grad(self.netD, False)
        # self.set_requires_grad(self.netE, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        self.set_requires_grad(self.netG, False)
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        if self.opt.display_aging_visuals:
            aging_visuals = {}
            for L in range(self.opt.num_classes):
                embedding = self.fixed_embeddings[L]
                aging_visuals[L] = self.netG(self.real_A[0:1, ...], embedding)
            for L in range(self.opt.num_classes):
                visual_ret['age_'+str(L)] = aging_visuals[L]
        self.set_requires_grad(self.netG, True)
        return visual_ret
