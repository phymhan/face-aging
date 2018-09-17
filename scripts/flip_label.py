import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='pairs.txt')
parser.add_argument('--ratio', type=float, default=0.1)
opt = parser.parse_args()

with open(opt.src, 'r') as f:
    sourcefile = [line.rstrip('\n') for line in f.readlines()]

with open(opt.src.replace('.txt', '_flip%d.txt'%(opt.ratio*100)), 'w') as f:
    for line in sourcefile:
        if random.random() < opt.ratio:
            label = int(line.split()[2])
            label_not = set([0, 1, 2]) - set([label])
            label_flipped = random.choice(list(label_not))
            line_new = ' '.join([line.split()[0], line.split()[1], str(label_flipped)])
        else:
            line_new = line
        f.write(line_new + '\n')
