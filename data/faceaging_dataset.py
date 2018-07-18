import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_faceaging_dataset
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
        self.img_dir = os.path.join(opt.dataroot)
        self.img_paths, self.ageList, self.size = make_faceaging_dataset(self.img_dir, self.opt)
        self.size = min(self.size, self.opt.max_dataset_size)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        imgs = {L: None for L in range(self.num_classes)}
        # pths = {L: '' for L in range(self.num_classes)}
        for L in range(self.num_classes):
            idx = index % len(self.ageList[L])
            id = self.ageList[L][idx]
            img = Image.open(self.img_paths[id]).convert('RGB')
            img = self.transform(img)
            if self.opt.input_nc == 1:  # RGB to gray
                tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)
            imgs[L] = img
            # pths[L] = self.img_paths[id]
        return imgs

    def __len__(self):
        # shuffle ageList
        self.shuffle_age_list()
        return self.size

    def shuffle_age_list(self):
        for L in range(self.num_classes):
            random.shuffle(self.ageList[L])

    def name(self):
        return 'FaceAgingDataset'
