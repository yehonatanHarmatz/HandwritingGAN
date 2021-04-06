import shutil
import random

from PIL import Image

from data.create_style_dataset import create_writers_dict
import os


def fix_image(imagePath):
    if not os.path.exists(imagePath):
        print('%s does not exist' % imagePath)
        return False
    try:
        im = Image.open(imagePath)
    except:
        return False
    # reshape the image to the new dimensions
    # im = im.resize((new_width, new_height))
    # append with 256 to add left, upper and lower white edges
    # init_w = int(random.normalvariate(init_gap, init_gap / 2))
    new_im = Image.new("RGB", (299, 299), color=(256, 256, 256))
    new_im.paste(im)
    im = new_im
    return im

def unpack_partition(new_dir, paths_list, pick=25000):
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
        os.makedirs(new_dir)
    else:
        os.makedirs(new_dir)
    new_paths_list = random.sample(paths_list, pick)
    for i, path in enumerate(new_paths_list):
        im = fix_image(path)
        im.save(os.path.join(new_dir,str(i)+'.png'))

if __name__ == '__main__':
    dataset = 'IAM'  # CVL/IAM/RIMES/gw
    mode = 'gan_test'  # tr/te/val/all
    # labeled = True
    top_dir = 'Datasets'
    pick = 10000
    # parameter relevant for IAM/RIMES:
    words = True  # use words images, otherwise use lines
    # parameters relevant for IAM:
    # author_number = -1  # use only images of a specific writer. If the value is -1, use all writers, otherwise use the index of this specific writer
    remove_punc = True  # remove images which include only one punctuation mark from the list ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']
    writers_images, _ = create_writers_dict(top_dir, dataset, mode, words, remove_punc)
    id = '2'
    dir_path = os.path.join(top_dir, mode + ' '+ str(pick), id)
    paths_to_move = [labeld_image[0] for writer_list in writers_images.values() for labeld_image in writer_list]
    unpack_partition(dir_path, paths_to_move, pick)