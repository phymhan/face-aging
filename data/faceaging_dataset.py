import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import get_age_label, get_age, get_age_label
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
        # sourcefile: filename <age (optional)>
        self.dir = self.root
        with open(opt.sourcefile_A, 'r') as f:
            sourcefile = f.readlines()
        self.sourcefile = [line.rstrip('\n') for line in sourcefile]
        self.create_age_list()
        self.size = min(self.size, self.opt.max_dataset_size)
        self.transform = get_transform(opt)

    def create_age_list(self):
        ageList = [[] for _ in range(self.num_classes)]  # list of list, the outer list is indexed by age label
        ages = []
        paths = []
        for (id, line) in enumerate(self.sourcefile, 0):
            line_splitted = line.split()
            if len(line_splitted) > 1:
                age = float(line_splitted[1])
            else:
                age = get_age(line_splitted[0])
            ages.append(age)
            age_label = get_age_label(age, self.age_bins_with_inf)
            ageList[age_label].append(id)
            paths.append(os.path.join(self.dir, line_splitted[0]))
        self.ageList = ageList
        self.ageListLen = [len(ls) for ls in ageList]
        self.size = max(self.ageListLen)
        self.ages = ages
        self.paths = paths

    def shuffle_age_list(self):
        for L in range(self.num_classes):
            random.shuffle(self.ageList[L])

    def __getitem__(self, index):
        ret_dict = {}
        for L in range(self.num_classes):
            idx = index % self.ageListLen[L]
            id = self.ageList[L][idx]
            img = Image.open(self.paths[id]).convert('RGB')
            img = self.transform(img)
            if self.opt.input_nc == 1:  # RGB to gray
                img = (img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114).unsqueeze(0)
            ret_dict[L] = img
            ret_dict['path_'+str(L)] = self.paths[id]
        return ret_dict

    def __len__(self):
        # shuffle ageList
        self.shuffle_age_list()
        return self.size

    def name(self):
        return 'FaceAgingDataset'
