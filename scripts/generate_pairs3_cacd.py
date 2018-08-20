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
    age = int(strlist[0])
    return age

# root = '/media/ligong/Toshiba/Datasets/CACD/CACD_cropped2_400'
mode = opt.mode
src = '../sourcefiles/CACD_'+mode+'_10k.txt'
N = opt.N

with open(src, 'r') as f:
    fnames = f.readlines()

fnames = [fname.rstrip('\n') for fname in fnames]

age_list = [[] for _ in range(5)]
for name in fnames:
    L = parse_age_label(name, [11, 21, 31, 41, 51, float('inf')])
    age_list[L].append(name)

# print(age_list)

def label_fn(a1, a2, m):
    if abs(a1-a2) <= m:
        return 1
    elif a1 < a2:
        return 0
    else:
        return 2

cnt = [0, 0, 0]
random.shuffle(fnames)
with open(mode+'_pairs_m10_cacd_10k_group.txt', 'w') as f:
    for _ in range(N):
        labels = random.sample(range(5), 2)
        name1 = random.choice(age_list[labels[0]])
        name2 = random.choice(age_list[labels[1]])
        label = 0 if labels[0] < labels[1] else 2
        cnt[label] += 1
        f.write('%s %s %d\n' % (name1, name2, label))

print(cnt)

# w = []
# for c in cnt:
#     w.append(1.0 * sum(cnt) / c)

# print([x/sum(w) for x in w])
