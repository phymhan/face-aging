from shutil import copyfile
import os


with open('/media/ligong/Toshiba/Datasets/SCUT-FBP/Rating_Collection/Attractiveness label.csv') as f:
    lines = f.readlines()

del lines[0]

# rename all files
for line in lines:
    line_splitted = line.split(',')
    src = os.path.join('/media/ligong/Toshiba/Datasets/SCUT-FBP/images_cropped_400', 'SCUT-FBP-'+line_splitted[0].rstrip(' ')+'.png')
    dst = os.path.join('/media/ligong/Toshiba/Datasets/SCUT-FBP/images_renamed', line_splitted[1].rstrip(' ')+'_'+line_splitted[0].rstrip(' ')+'.png')
    copyfile(src, dst)
