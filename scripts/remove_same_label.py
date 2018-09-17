src = '../sourcefiles/train_pairs_m20_cacd_10k.txt'
dst = '../sourcefiles/train_diff_pairs_m20_cacd_10k.txt'

with open(src, 'r') as f:
    lines = f.readlines()

with open(dst, 'w') as f:
    for line in lines:
        label = int(line.split()[2])
        if label != 1:
            f.write(line)
