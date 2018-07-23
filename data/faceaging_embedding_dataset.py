import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import parse_age_label
from PIL import Image
import random


# TODO: set random seed
class FaceAgingEmbeddingDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # labels: A < B: 0, A = B: 1, A > B: 2
        self.dir = self.root
        with open(opt.sourcefile_A, 'r') as f:
            sourcefile = f.readlines()
        for (i, line) in enumerate(sourcefile, 0):
            sourcefile[i] = line.rstrip('\n')
        self.sourcefile = sourcefile
        self.size = min(len(self.sourcefile), self.opt.max_dataset_size)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        line = self.sourcefile[index].split()
        A_path = os.path.join(self.dir, line[0])
        B_path = os.path.join(self.dir, line[1])
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc
        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'label': int(line[2]),
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        # shuffle sourcefile
        random.shuffle(self.sourcefile)
        return self.size

    def name(self):
        return 'FaceAgingEmbeddingDataset'
