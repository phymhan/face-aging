import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import parse_age_label
from PIL import Image
import random


# TODO: set random seed
class FaceAgingEmbeddingVisDataset(BaseDataset):
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

        self.num_classes = len(opt.age_binranges)
        self.age_bins = opt.age_binranges
        self.age_bins_with_inf = opt.age_binranges + [float('inf')]
        with open(opt.sourcefile_B, 'r') as f:
            sourcefile = f.readlines()
        self.paths = [os.path.join(self.dir, name.rstrip('\n').split()[0]) for name in sourcefile]
        self.fnames = [name.rstrip('\n').split()[0] for name in sourcefile]
        # parse paths
        ageList = [[] for _ in range(self.num_classes)]  # list of list, the outer list is indexed by age label
        for (id, fname) in enumerate(self.fnames, 0):
            L = parse_age_label(fname, self.age_bins_with_inf)
            ageList[L].append(id)
        self.ageList = ageList

    def __getitem__(self, index):
        # Siamese pairs
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
        ret_dict = {'A': A, 'B': B, 'label': int(line[2]), 'A_paths': A_path, 'B_paths': B_path}

        # Aging visual samples
        for L in range(self.num_classes):
            idx = index % len(self.ageList[L])
            id = self.ageList[L][idx]
            img = Image.open(self.paths[id]).convert('RGB')
            img = self.transform(img)
            if self.opt.input_nc == 1:  # RGB to gray
                tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)
            ret_dict[L] = img
            ret_dict['path_' + str(L)] = self.paths[id]
        return ret_dict

    def __len__(self):
        # shuffle sourcefile
        random.shuffle(self.sourcefile)
        return self.size

    def name(self):
        return 'FaceAgingEmbeddingVisDataset'
