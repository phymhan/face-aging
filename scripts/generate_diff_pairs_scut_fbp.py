import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--margin', type=float, default=0.35)
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


root = '/media/ligong/Toshiba/Datasets/SCUT-FBP/images_renamed'
mode = opt.mode
# src = '../data/'+mode+'_scut-fbp.txt'
N = opt.N

# with open(src, 'r') as f:
#     fnames = f.readlines()
fnames = os.listdir(root)

score_list = [[] for _ in range(5)]
for name in fnames:
    score = parse_score(name)
    score_list[parse_score_label(score)].append(name)

print(score_list)

with open(mode+'_diff_pairs_scut-fbp.txt', 'w') as f:
    for _ in range(N):
        labels = random.sample(range(5), 2)
        name1 = random.choice(score_list[labels[0]])
        name2 = random.choice(score_list[labels[1]])
        label = label_fn(parse_score(name1), parse_score(name2), opt.margin)
        f.write('%s %s %d\n' % (name1, name2, label))
