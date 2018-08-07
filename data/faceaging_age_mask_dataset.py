import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import parse_age_label, parse_age
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

        ageA = torch.Tensor([parse_age(fnameA)]).reshape(1, 1, 1)
        ageB = torch.Tensor([parse_age(fnameB)]).reshape(1, 1, 1)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        idA = fnameA.split('_')[1]
        idB = fnameB.split('_')[1]

        maskA = Image.open(os.path.join(self.root_mask, idA)).convert('RGB')
        maskB = Image.open(os.path.join(self.root_mask, idB)).convert('RGB')
        maskA = self.transform(maskA)
        maskB = self.transform(maskB)

        return {'A': imgA, 'B': imgB, 'A_mask': maskA, 'B_mask': maskB, 'A_age': ageA, 'B_age': ageB, 'label': int(line[2]), 'B_paths': B_path}

    def __len__(self):
        return len(self.sourcefile)

    def name(self):
        return 'FaceAgingAgeMaskDataset'
