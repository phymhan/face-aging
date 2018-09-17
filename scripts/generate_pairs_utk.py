# datafile: A B 0/1/2
# label: 0: A < B, 1: A == B, 2: A > B

import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--N', type=int, default=140000)
parser.add_argument('--margin', type=int, default=10)
parser.add_argument('--noise', type=float, default=5)
opt = parser.parse_args()

def parse_age_label(fname, binranges):
    strlist = fname.split('_')
    age = int(strlist[0])
    l = None
    for l in range(len(binranges)-1):
        if (age >= binranges[l]) and (age < binranges[l+1]):
            break
    return l

def parse_age(fname):
    strlist = fname.split('_')
    age = float(strlist[0])
    return age

def label_fn(a1, a2, m):
    if abs(a1-a2) <= m:
        return 1
    elif a1 < a2:
        return 0
    else:
        return 2

# generate a LUT for noisy labels
# generate noisy pairs according to LUT

# root = '/media/ligong/Toshiba/Datasets/CACD/CACD_cropped2_400'
mode = opt.mode
src = '../sourcefiles/utk_'+mode+'.txt'
N = opt.N

with open(src, 'r') as f:
    fnames = f.readlines()

fnames = [fname.rstrip('\n') for fname in fnames]
fnames_noisy = {}

# LUT
with open('utk_'+mode+'_noisy%d.txt'%opt.noise, 'w') as f:
    for name in fnames:
        s = name.split('_')
        age = float(s[0])
        age_noisy = max(0, age + random.uniform(-opt.noise, opt.noise))
        fnames_noisy[name] = age_noisy
        f.write(name + ' %.2f'%age_noisy + '\n')

# pairs
random.shuffle(fnames)
with open(mode+'_pairs_m%d_utk.txt'%opt.margin, 'w') as f, open(mode+'_pairs_m%d_utk_noisy%d.txt'%(opt.margin, opt.noise), 'w') as fn:
    for _ in range(N):
        ss = random.sample(fnames, 2)
        name1 = ss[0]
        name2 = ss[1]
        label = label_fn(parse_age(name1), parse_age(name2), opt.margin)
        f.write('%s %s %d\n' % (name1, name2, label))

        age1n = fnames_noisy[name1]
        age2n = fnames_noisy[name2]
        labeln = label_fn(age1n, age2n, opt.margin)
        fn.write('%s %s %d\n' % (name1, name2, labeln))
