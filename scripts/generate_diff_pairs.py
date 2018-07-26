# datafile: A B 0/1/2
# label: 0: A < B, 1: A == B, 2: A > B

import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--N', type=int, default=24000)
opt = parser.parse_args()

def parse_age_label(fname, binranges):
    strlist = fname.split('_')
    age = int(strlist[0])
    l = None
    for l in range(len(binranges)-1):
        if (age >= binranges[l]) and (age < binranges[l+1]):
            break
    return l

root = '/media/ligong/Toshiba/Datasets/UTKFace'
mode = opt.mode
src = '../data/UTK_'+mode+'.txt'
N = opt.N
binranges = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] + [float('inf')]
binranges = [1, 21, 41, 61, 81] + [float('inf')]
num_classes = len(binranges)-1

paths = [[] for _ in range(num_classes)]
with open(src, 'r') as f:
    fnames = f.readlines()
for id, fname in enumerate(fnames):
    fname = fname.rstrip('\n').split()[0]
    label = parse_age_label(fname, binranges)
    paths[label].append(fname)
    print('--> %s %d' % (fname, label))

def label_fn(l1, l2):
    if l1 < l2:
        return 0
    elif l1 == l2:
        return 1
    else:
        return 2

with open(mode+'_diff_pairs.txt', 'w') as f:
    for _ in range(N):
        l1 = random.choice(range(num_classes))
        l2 = random.sample(set(range(num_classes))-set([l1]), 1)[0]
        path1 = random.choice(paths[l1])
        path2 = random.choice(paths[l2])
        label = label_fn(l1, l2)
        f.write('%s %s %d\n' % (path1, path2, label))

