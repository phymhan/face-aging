src = '../sourcefiles/train_pairs_m10_cacd_10k.txt'
dst = '../sourcefiles/train_pairs_m20_cacd_10k.txt'

margin = 20

def label_fn(a1, a2, m):
    if abs(a1-a2) <= m:
        return 1
    elif a1 < a2:
        return 0
    else:
        return 2


def parse_age(fname):
    strlist = fname.split('_')
    age = int(strlist[0])
    return age


with open(src, 'r') as f:
    lines = f.readlines()

with open(dst, 'w') as f:
    for line in lines:
        name1 = line.split()[0]
        name2 = line.split()[1]
        age1 = parse_age(name1)
        age2 = parse_age(name2)
        label = label_fn(age1, age2, margin)
        f.write('%s %s %d\n' % (name1, name2, label))
