import torch
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from util.util import upsample2d, expand2d, expand2d_as
from .base_model import BaseModel
from . import networks
import random
import numpy as np
from collections import OrderedDict


# TODO: set random seed
class FaceAgingModel(BaseModel):
    def name(self):
        return 'FaceAgingModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        parser.add_argument('--norm_G', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='batch', help='instance normalization or batch normalization')
        parser.add_argument('--embedding_nc', type=int, default=10, help='# of embedding channels')
        parser.add_argument('--display_aging_visuals', action='store_true', help='display aging visuals if True')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=0.001, help='weight for L1 loss')
            parser.add_argument('--lambda_IP', type=float, default=0.1, help='weight for identity preserving loss')
            parser.add_argument('--lambda_AC', type=float, default=1, help='weight for auxiliary classifier')
            parser.add_argument('--lambda_A', type=float, default=1, help='weight for cycle consistency loss')
            parser.add_argument('--lambda_A_GAN', type=float, default=1, help='weight for GAN loss on rec_A')
            parser.add_argument('--which_model_netIP', type=str, default='alexnet', help='model type for IP loss')
            parser.add_argument('--which_model_netAC', type=str, default='alexnet', help='model type for AC loss')
            parser.add_argument('--pooling_AC', type=str, default='avg', help='which pooling layer in AC')
            parser.add_argument('--dropout_AC', type=float, default=0.5, help='p in dropout layer')
            parser.add_argument('--fineSize_IP', type=int, default=224, help='fineSize for IP')
            parser.add_argument('--fineSize_AC', type=int, default=224, help='fineSize for AC')
            parser.add_argument('--pretrained_model_path_IP', type=str, default='pretrained_models/alexnet.pth', help='pretrained model path to IP net')
            parser.add_argument('--pretrained_model_path_AC', type=str, default='pretrained_models/alexnet.pth', help='pretrained model path to AC net')
            parser.add_argument('--train_label_pairs', type=str, default='', help='file path of train label pairs')
            parser.add_argument('--lr_AC', type=float, default=0.0002, help='learning rate for AC')
            parser.add_argument('--train_aux_on_fake', action='store_true', help='if True, train AC on fake images')
            parser.add_argument('--no_trick', action='store_true')
            parser.add_argument('--identity_preserving_criterion', type=str, default='mse', help='which criterion to use for identity preserving loss')
            parser.add_argument('--detach_fake_B', action='store_true', help='if True, detach fake_B when computing rec_A')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.input_nc == opt.output_nc)
        self.opt.num_classes = len(opt.age_binranges)
        assert(opt.embedding_nc == self.opt.num_classes)

        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_GAN_cycle', 'G_IP', 'G_L1', 'G_AC', 'G_cycle',
                           'D_real_right', 'D_real_wrong', 'D_fake', 'AC_real', 'AC_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'rec_A']
        else:
            self.visual_names = ['real_A']
        if self.isTrain:
            self.model_names = ['G', 'D', 'AC']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.embedding_nc, opt.ngf,
                                      which_model_netG=opt.which_model_netG,
                                      norm=opt.norm_G, nl=opt.nl, dropout=opt.dropout, init_type=opt.init_type,
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
            # define netAC
            self.netAC = networks.define_AC(opt.which_model_netAC, opt.input_nc, opt.init_type, opt.num_classes,
                                            pooling=opt.pooling_AC, dropout=opt.dropout_AC, gpu_ids=self.gpu_ids)
            if not opt.continue_train and opt.pretrained_model_path_AC:
                if isinstance(self.netAC, torch.nn.DataParallel):
                    self.netAC.module.load_state_dict(torch.load(opt.pretrained_model_path_AC, map_location=str(self.device)), strict=True)
                else:
                    self.netAC.load_state_dict(torch.load(opt.pretrained_model_path_AC, map_location=str(self.device)), strict=True)

        if self.isTrain:
            # TODO: use num_classes pools
            self.fake_B_pool = [ImagePool(opt.pool_size) for _ in range(self.opt.num_classes)]
            # define loss functions
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            if opt.identity_preserving_criterion.lower() == 'mse':
                self.criterionIP = torch.nn.MSELoss()
            elif opt.identity_preserving_criterion.lower() == 'l1':
                self.criterionIP = torch.nn.L1Loss()
            else:
                raise NotImplementedError('Not Implemented')
            self.criterionAC = torch.nn.CrossEntropyLoss()
            self.criterionCycle = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AC = torch.optim.Adam(self.netAC.parameters(), lr=opt.lr_AC, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_AC)

        if self.isTrain:
            if self.opt.train_label_pairs:
                with open(self.opt.train_label_pairs, 'r') as f:
                    train_label_pairs = f.readlines()
                self.train_label_pairs = [line.rstrip('\n') for line in train_label_pairs]
            else:
                self.train_label_pairs = None

        self.pre_generate_embeddings()

        if self.isTrain:
            self.transform_IP = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).to(self.device)
            self.transform_AC = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).to(self.device)

    def pre_generate_embeddings(self):
        one_hot_labels = []
        for L in range(self.opt.num_classes):
            one_hot_vector = np.zeros(self.opt.num_classes)
            one_hot_vector[L] = 1
            one_hot_labels.append(torch.Tensor(one_hot_vector).view(1, self.opt.num_classes, 1, 1).to(self.device))
        self.one_hot_labels = one_hot_labels

    def sample_labels(self):
        # returns label_A, label_B, label_B_not
        if self.train_label_pairs:
            idx = self.current_iter % len(self.train_label_pairs)
            labels_AnB = self.train_label_pairs[idx].split()
        else:
            labels_AnB = random.sample(range(self.opt.num_classes), 2)
        return int(labels_AnB[0]), int(labels_AnB[1]), random.sample(set(range(self.opt.num_classes))-set([int(labels_AnB[1])]), 1)[0]

    def set_input(self, input):
        if self.isTrain:
            self.label_A, self.label_B, self.label_B_not = self.sample_labels()
            self.real_A = input[self.label_A].to(self.device)
            self.real_B = input[self.label_B].to(self.device)
            self.real_A_IP = upsample2d(self.real_A, self.opt.fineSize_IP)
            self.real_B_AC = upsample2d(self.real_B, self.opt.fineSize_AC)
            self.image_paths = input['path_'+str(self.label_B)]
        else:
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']
        self.current_iter += 1
        self.current_batch_size = int(self.real_A.size(0))

    def forward(self):
        self.fake_B = self.netG(self.real_A, self.one_hot_labels[self.label_B])
        self.fake_B_IP = upsample2d(self.fake_B, self.opt.fineSize_IP)
        self.fake_B_AC = upsample2d(self.fake_B, self.opt.fineSize_AC)
        if not self.opt.detach_fake_B:
            self.rec_A = self.netG(self.fake_B, self.one_hot_labels[self.label_A])
        else:
            self.rec_A = self.netG(self.fake_B.detach(), self.one_hot_labels[self.label_A])

    def test(self):
        return

    def backward_D(self):
        # Fake image with label_B
        # stop backprop to the generator by detaching fake_B
        fake_B = torch.cat((self.fake_B_pool[self.label_B].query(self.fake_B),
                            expand2d_as(self.one_hot_labels[self.label_B], self.real_A)), 1)
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real image with label_B
        real_B_right = torch.cat((self.real_B, expand2d_as(self.one_hot_labels[self.label_B], self.real_A)), 1)
        pred_real = self.netD(real_B_right)
        self.loss_D_real_right = self.criterionGAN(pred_real, True)

        if not self.opt.no_trick:
            # Real image with label_B_not
            real_B_wrong = torch.cat((self.real_B, expand2d_as(self.one_hot_labels[self.label_B_not], self.real_A)), 1)
            pred_real = self.netD(real_B_wrong)
            self.loss_D_real_wrong = self.criterionGAN(pred_real, False)

            # Combined loss
            self.loss_D = (self.loss_D_fake + (self.loss_D_real_right + self.loss_D_real_wrong) * 0.5) * 0.5
        else:
            self.loss_D_real_wrong = 0
            self.loss_D = (self.loss_D_fake + self.loss_D_real_right) * 0.5

        self.loss_D.backward()

    def backward_AC(self):
        # Real
        pred = self.netAC(self.transform_AC(self.real_B_AC))
        self.loss_AC_real = self.criterionAC(pred, torch.LongTensor([self.label_B]).expand(self.real_A.size(0)).to(self.device))

        # Fake
        if self.opt.train_aux_on_fake:
            pred = self.netAC(self.transform_AC(self.fake_B_AC.detach()))
            self.loss_AC_fake = self.criterionAC(pred, torch.LongTensor([self.label_B]).expand(self.real_A.size(0)).to(self.device))
        else:
            self.loss_AC_fake = 0.0

        self.loss_AC = (self.loss_AC_fake + self.loss_AC_real) * 0.5

        self.loss_AC.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = torch.cat((self.fake_B, expand2d_as(self.one_hot_labels[self.label_B], self.real_A)), 1)
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN loss on rec_A
        if self.opt.lambda_A_GAN > 0.0:
            fake_A = torch.cat((self.rec_B, expand2d_as(self.one_hot_labels[self.label_A], self.real_A)), 1)
            pred_fake = self.netD(fake_A)
            self.loss_G_GAN_cycle = self.criterionGAN(pred_fake, True) * self.opt.lambda_A_GAN
        else:
            self.loss_G_GAN_cycle = 0.0

        # L1: fake_B ~= real_A
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        # IP loss
        if self.opt.lambda_IP > 0.0:
            feature_A = self.netIP(self.transform_IP(self.real_A_IP)).detach()
            feature_A.requires_grad = False
            self.loss_G_IP = self.criterionIP(self.netIP(self.transform_IP(self.fake_B_IP)), feature_A) * self.opt.lambda_IP
        else:
            self.loss_G_IP = 0.0

        # AC loss
        if self.opt.lambda_AC > 0.0:
            pred_fake = self.netAC(self.transform_AC(self.fake_B_AC))
            self.loss_G_AC = self.criterionAC(pred_fake, torch.LongTensor([self.label_B]).expand(self.real_A.size(0)).to(self.device)) * self.opt.lambda_AC
        else:
            self.loss_G_AC = 0.0

        # Cycle loss
        if self.opt.lambda_A > 0.0:
            self.loss_G_cycle = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        else:
            self.loss_G_cycle = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_IP + self.loss_G_L1 + self.loss_G_AC + self.loss_G_cycle + self.loss_G_GAN_cycle

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update AC
        self.set_requires_grad(self.netAC, True)
        self.optimizer_AC.zero_grad()
        self.backward_AC()
        self.optimizer_AC.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netAC, False)
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
            aging_visuals = {L: self.netG(self.real_A[0:1, ...], self.one_hot_labels[L]) for L in range(self.opt.num_classes)}
            for L in range(self.opt.num_classes):
                visual_ret['age_'+str(L)] = aging_visuals[L]
        self.set_requires_grad(self.netG, True)
        return visual_ret
