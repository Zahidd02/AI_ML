# first run the below code to create directories in Colab
# !mkdir /content/images
# !unzip zipped_images_file -d /content/images/all
# !mkdir /content/images/train; mkdir /content/images/validation; mkdir /content/images/test

import glob
from pathlib import Path
import random
import os

image_path = '/content/images/all'
train_path = '/content/images/train'
val_path = '/content/images/validation'
test_path = '/content/images/test'

jpg_file_list = [path for path in Path(image_path).rglob('*.jpg')]
JPG_file_list = [path for path in Path(image_path).rglob('*.JPG')]
png_file_list = [path for path in Path(image_path).rglob('*.png')]
bmp_file_list = [path for path in Path(image_path).rglob('*.bmp')]

file_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list
file_num = len(file_list)
print('Total images: %d' % file_num)

train_percent = 0.8
val_percent = 0.1
test_percent = 0.1
train_num = int(file_num * train_percent)
val_num = int(file_num * val_percent)
test_num = file_num - train_num - val_num
print('Images moving to train: %d' % train_num)
print('Images moving to validation: %d' % val_num)
print('Images moving to test: %d' % test_num)

for i in range(train_num):
    move_me = random.choice(file_list)
    fn = move_me.name
    base_fn = move_me.stem
    parent_path = move_me.parent
    xml_fn = base_fn + '.xml'
    os.rename(move_me, train_path + '/' + fn)
    os.rename(os.path.join(parent_path, xml_fn), os.path.join(train_path, xml_fn))
    file_list.remove(move_me)

for i in range(val_num):
    move_me = random.choice(file_list)
    fn = move_me.name
    base_fn = move_me.stem
    parent_path = move_me.parent
    xml_fn = base_fn + '.xml'
    os.rename(move_me, val_path + '/' + fn)
    os.rename(os.path.join(parent_path, xml_fn), os.path.join(val_path, xml_fn))
    file_list.remove(move_me)

for i in range(test_num):
    move_me = random.choice(file_list)
    fn = move_me.name
    base_fn = move_me.stem
    parent_path = move_me.parent
    xml_fn = base_fn + '.xml'
    os.rename(move_me, test_path + '/' + fn)
    os.rename(os.path.join(parent_path, xml_fn), os.path.join(test_path, xml_fn))
    file_list.remove(move_me)
