import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
import numpy as np


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
        parser.add_argument('--embedding_nc', type=int, default=10, help='# of embedding channels')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=0.001, help='weight for L1 loss')
            parser.add_argument('--lambda_IP', type=float, default=0.1, help='weight for identity preserving loss')
            parser.add_argument('--lambda_AC', type=float, default=1, help='weight for auxiliary classifier')
            parser.add_argument('--which_model_netIP', type=str, default='alexnet', help='model type for IP loss')
            parser.add_argument('--which_model_netAC', type=str, default='alexnet', help='model type for AC loss')
            parser.add_argument('--IP_pretrained_model_path', type=str, default='pretrained_models/alexnet.pth', help='pretrained model path to IP net')
            parser.add_argument('--AC_pretrained_model_path', type=str, default='pretrained_models/alexnet.pth', help='pretrained model path to AC net')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.input_nc == opt.output_nc)
        self.opt.num_classes = len(opt.age_binranges)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_IP', 'G_L1', 'G_AC', 'D_real', 'D_fake', 'AC_real', 'AC_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D', 'AC']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc + opt.embedding_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.embedding_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netIP = networks.define_IP(opt.which_model_netIP, opt.input_nc, self.gpu_ids)
            self.netIP.load_state_dict(torch.load(opt.IP_pretrained_model_path, map_location=str(self.device)), strict=False)
            self.netAC = networks.define_AC(opt.which_model_netAC, opt.input_nc, opt.num_classes, opt.init_type, self.gpu_ids)
            if not opt.continue_train and opt.AC_pretrained_model_path:
                state_dict = torch.load(opt.AC_pretrained_model_path, map_location=str(self.device))
                if opt.which_model_netAC == 'alexnet':
                    del state_dict['classifier.6.weight']
                    del state_dict['classifier.6.bias']
                self.netAC.load_state_dict(state_dict, strict=False)

        if self.isTrain:
            # TODO: use num_classes pools
            assert(opt.pool_size == 0)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionAC = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AC = torch.optim.Adam(self.netAC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_AC)

        if self.isTrain:
            if not self.opt.train_label_pairs:
                with open(self.opt.train_label_pairs, 'r') as f:
                    self.opt.train_label_pairs = f.readlines()

        self.pre_generate_embeddings()
        self.pre_generate_labels()

    def pre_generate_embeddings(self):
        one_hot_labels = []
        one = np.ones((self.opt.fineSize, self.opt.fineSize))

        # for one sample
        for L in range(self.opt.num_classes):
            tmp = np.zeros((self.opt.num_classes, self.opt.fineSize, self.opt.fineSize))
            tmp[L, :, :] = one
            one_hot_labels.append(torch.Tensor(tmp))

        # for a batch
        batch_one_hot_labels = []
        for L in range(self.opt.num_classes):
            tmp_one_hot_label = np.zeros((self.opt.batchSize, self.opt.num_classes, self.opt.fineSize, self.opt.fineSize))
            for j in range(self.opt.batchSize):
                tmp_one_hot_label[j, :, :, :] = one_hot_labels[L]
                batch_one_hot_labels.append(torch.Tensor(tmp_one_hot_label).to(self.device))

        self.one_hot_labels = one_hot_labels
        self.batch_one_hot_labels = batch_one_hot_labels

    def pre_generate_labels(self):
        batch_labels = []
        for L in range(self.opt.num_classes):
            tmp_labels = np.ones((self.opt.batchSize), dtype=np.int) * L
            batch_labels.append(torch.LongTensor(tmp_labels).to(self.device))

        self.batch_labels = batch_labels

    def sample_labels(self):
        if not self.opt.train_label_pairs:
            idx = self.current_iter % len(self.opt.train_label_pairs)
            labels_AnB = self.opt.train_label_pairs[idx].rstrip('\n').split()
        else:
            labels_AnB = random.sample(range(self.opt.num_classes), 2)
        return int(labels_AnB[0]), int(labels_AnB[1])

    def set_input(self, input):
        self.label_A, self.label_B = self.sample_labels()
        self.real_A = input[self.label_A].to(self.device)
        self.real_B = input[self.label_B].to(self.device)
        self.image_paths = ''  # TODO: add image paths
        self.current_iter += 1

    def forward(self):
        self.fake_B = self.netG(torch.cat((self.real_A, self.batch_one_hot_labels[self.label_B][0:self.real_A.size(0),...]), 1))

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B_pool.query(torch.cat((self.fake_B, self.batch_one_hot_labels[self.label_B][0:self.real_A.size(0),...]), 1))
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_B = torch.cat((self.real_B, self.batch_one_hot_labels[self.label_B][0:self.real_A.size(0),...]), 1)
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_AC(self):
        # Real
        pred = self.netAC(self.real_B)
        self.loss_AC_real = self.criterionAC(pred, self.batch_labels[self.label_B][0:self.real_A.size(0),...])

        # Fake
        pred = self.netAC(self.fake_B.detach())
        self.loss_AC_fake = self.criterionAC(pred, self.batch_labels[self.label_B][0:self.real_A.size(0),...])

        self.loss_AC = (self.loss_AC_fake + self.loss_AC_real) * 0.5

        self.loss_AC.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = torch.cat((self.fake_B, self.batch_one_hot_labels[self.label_B][0:self.real_A.size(0),...]), 1)
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1: fake_B ~= real_A
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1

        # IP loss
        feature_A = self.netIP(self.real_A).detach()
        feature_A.requires_grad = False
        self.loss_G_IP = torch.nn.MSELoss()(self.netIP(self.fake_B), feature_A) * self.opt.lambda_IP

        # AC loss
        pred_fake = self.netAC(self.fake_B)
        self.loss_G_AC = self.criterionAC(pred_fake, self.batch_labels[self.label_B][0:self.real_A.size(0),...]) * self.opt.lambda_AC

        self.loss_G = self.loss_G_GAN + self.loss_G_IP + self.loss_G_L1

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
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
