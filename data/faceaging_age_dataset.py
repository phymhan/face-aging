import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import parse_age_label, parse_age
from PIL import Image
import random
import torch


# TODO: set random seed
class FaceAgingAgeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.sourcefile = opt.sourcefile_A
        self.transform = get_transform(opt)
        with open(self.sourcefile, 'r') as f:
            self.sourcefile = f.readlines()

    def __getitem__(self, index):
        line = self.sourcefile[index].rstrip('\n').split()
        fnameA, fnameB = line[0], line[1]
        A_path = os.path.join(self.root, fnameA)
        B_path = os.path.join(self.root, fnameB)
        imgA = Image.open(A_path).convert('RGB')
        imgB = Image.open(B_path).convert('RGB')

        labelA = torch.Tensor([parse_age(fnameA)]).reshape(1, 1, 1)
        labelB = torch.Tensor([parse_age(fnameB)]).reshape(1, 1, 1)
        # labelA = torch.Tensor([(parse_age_label(fnameA, self.opt.age_binranges + [float('inf')])+0)*20]).reshape(1, 1, 1)
        # labelB = torch.Tensor([(parse_age_label(fnameB, self.opt.age_binranges + [float('inf')])+0)*20]).reshape(1, 1, 1)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return {'A': imgA, 'B': imgB, 'A_label': labelA, 'B_label': labelB, 'label': int(line[2]), 'B_paths': B_path}

    def __len__(self):
        return len(self.sourcefile)

    def name(self):
        return 'FaceAgingAgeDataset'
