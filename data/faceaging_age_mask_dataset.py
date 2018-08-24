import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import get_age_label, get_age
from PIL import Image
import random
import torch


# TODO: set random seed
class FaceAgingAgeMaskDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        with open(opt.sourcefile_A, 'r') as f:
            sourcefile = f.readlines()
        self.sourcefile = [line.rstrip('\n') for line in sourcefile]
        self.transform = get_transform(opt)

        self.root_mask = opt.dataroot_mask

    def __getitem__(self, index):
        line = self.sourcefile[index].split()
        fnameA, fnameB = line[0], line[1]
        A_path = os.path.join(self.root, fnameA)
        B_path = os.path.join(self.root, fnameB)
        imgA = Image.open(A_path).convert('RGB')
        imgB = Image.open(B_path).convert('RGB')

        ageA = torch.Tensor([get_age(fnameA)]).reshape(1, 1, 1)
        ageB = torch.Tensor([get_age(fnameB)]).reshape(1, 1, 1)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        if self.opt.input_nc == 1:  # RGB to gray
            imgA = (imgA[0, ...] * 0.299 + imgA[1, ...] * 0.587 + imgA[2, ...] * 0.114).unsqueeze(0)
        if self.opt.output_nc == 1:
            imgB = (imgB[0, ...] * 0.299 + imgB[1, ...] * 0.587 + imgB[2, ...] * 0.114).unsqueeze(0)

        idA = fnameA.split('_')[1]
        idB = fnameB.split('_')[1]

        maskA = Image.open(os.path.join(self.root_mask, idA)).convert('RGB')
        maskB = Image.open(os.path.join(self.root_mask, idB)).convert('RGB')
        maskA = self.transform(maskA)
        maskB = self.transform(maskB)
        if self.opt.input_nc == 1:  # RGB to gray
            maskA = maskA[0:1, ...]
        if self.opt.output_nc == 1:
            maskB = maskB[0:1, ...]

        return {'A': imgA, 'B': imgB, 'A_mask': maskA, 'B_mask': maskB, 'A_age': ageA, 'B_age': ageB, 'label': int(line[2]), 'B_paths': B_path}

    def __len__(self):
        return len(self.sourcefile)

    def name(self):
        return 'FaceAgingAgeMaskDataset'
