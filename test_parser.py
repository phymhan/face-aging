import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', nargs='*', type=float, default=[0])

opt = parser.parse_args()

print(opt)
