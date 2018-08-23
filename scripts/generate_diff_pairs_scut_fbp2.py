import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--margin', type=float, default=0.5)
opt = parser.parse_args()


def parse_score_label(score):
    return round(score)-1


def parse_score(fname):
    strlist = fname.split('_')
    score = float(strlist[0])
    return score

def label_fn(a1, a2, m):
    if abs(a1-a2) <= m:
        return 1
    elif a1 < a2:
        return 0
    else:
        return 2


root = '../datasets/SCUT-FBP/'
mode = opt.mode
src = '../sourcefiles/scut-fbp_'+mode+'.txt'
N = opt.N

with open(src, 'r') as f:
    fnames = [fname.rstrip('\n') for fname in f.readlines()]
# fnames = os.listdir(root)

score_list = [[] for _ in range(5)]
for name in fnames:
    score = parse_score(name)
    score_list[parse_score_label(score)].append(name)

# print(score_list)

with open(mode+'_diff_pairs_scut-fbp2.txt', 'w') as f:
    for _ in range(N):
        name1 = random.choice(fnames)
        label1 = parse_score_label(parse_score(name1))
        name2 = random.choice(fnames)
        label = label_fn(parse_score(name1), parse_score(name2), opt.margin)
        if label != 1:
            f.write('%s %s %d\n' % (name1, name2, label))
