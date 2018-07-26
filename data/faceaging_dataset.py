import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import parse_age_label
from PIL import Image
import random


# TODO: set random seed
class FaceAgingDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        # opt.age_binranges: the (i+1)-th group is in the range [age_binranges[i], age_binranges[i+1])
        # e.g.: [1, 11, 21, ..., 101], the 1-st group is [1, 10], the 9-th [91, 100], however, the 10-th [101, +inf)
        self.opt = opt
        self.num_classes = len(opt.age_binranges)
        self.age_bins = opt.age_binranges
        self.age_bins_with_inf = opt.age_binranges + [float('inf')]
        self.root = opt.dataroot
        if not opt.sourcefile_A:
            self.dir = os.path.join(opt.dataroot, opt.phase)
            self.paths, self.fnames = make_dataset_with_filenames(self.dir)
        else:
            self.dir = self.root
            with open(opt.sourcefile_A, 'r') as f:
                sourcefile = f.readlines()
            self.paths = [os.path.join(self.dir, name.rstrip('\n').split()[0]) for name in sourcefile]
            self.fnames = [name.rstrip('\n').split()[0] for name in sourcefile]
        self.parse_paths()
        self.size = min(self.size, self.opt.max_dataset_size)
        self.transform = get_transform(opt)

    def parse_paths(self):
        ageList = [[] for _ in range(self.num_classes)]  # list of list, the outer list is indexed by age label
        for (id, fname) in enumerate(self.fnames, 0):
            L = parse_age_label(fname, self.age_bins_with_inf)
            ageList[L].append(id)
        maxLen = max([len(ls) for ls in ageList])
        self.ageList = ageList
        self.size = maxLen

    def __getitem__(self, index):
        ret_dict = {}
        for L in range(self.num_classes):
            idx = index % len(self.ageList[L])
            id = self.ageList[L][idx]
            img = Image.open(self.paths[id]).convert('RGB')
            img = self.transform(img)
            if self.opt.input_nc == 1:  # RGB to gray
                tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)
            ret_dict[L] = img
            ret_dict['path_'+str(L)] = self.paths[id]
        return ret_dict

    def __len__(self):
        # shuffle ageList
        self.shuffle_age_list()
        return self.size

    def shuffle_age_list(self):
        for L in range(self.num_classes):
            random.shuffle(self.ageList[L])

    def name(self):
        return 'FaceAgingDataset'
