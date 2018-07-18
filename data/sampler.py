from __future__ import absolute_import

from collections import defaultdict

import numpy as np
import random
import torch
from torch.utils.data.sampler import (RandomSampler, Sampler,
                                      SequentialSampler, SubsetRandomSampler,
                                      WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.dataMap = self.data_source.getMapINameIndex()
        self.identities = self.data_source.identities
        self.num_indentities = len(self.identities)

    def __len__(self):
        return self.num_indentities * 3

    def __iter__(self):
        pNames = list(self.identities.keys())
        random.shuffle(pNames)
        ret = []
        for i in range(len(pNames)):
            pName = pNames[i]
            ages = self.identities[pName]
            choAges = map(int, random.sample(ages.keys(), 3))
            choAges = sorted(choAges)
            for j in range(3):
                imageName = random.choice(ages[str(choAges[j])])
                ret.append(int(self.dataMap[imageName]))
        return iter(ret)


# Adapted from Brother Cheng's RandomIdentitySampler
# TODO: iter a list based on batch_size, this sampler is incomplete
class FaceAgingSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return None