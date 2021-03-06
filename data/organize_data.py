import shutil
import random

from PIL import Image

from data.create_style_dataset import create_writers_dict
import os


def fix_image(im):
    # reshape the image to the new dimensions
    # im = im.resize((new_width, new_height))
    # append with 256 to add left, upper and lower white edges
    # init_w = int(random.normalvariate(init_gap, init_gap / 2))
    new_im = Image.new("RGB", (299, 299), color=(256, 256, 256))
    new_im.paste(im)
    im = new_im
    return im

def resize_im(im, label, resize='charResize', imgH=32, h_gap=0, charminW=16, charmaxW=17, discard_wide=True, discard_narr=True):
    if resize in ['charResize', 'keepRatio']:
        width, height = im.size
        new_height = imgH - (h_gap * 2)
        len_word = len(label)
        width = int(width * imgH / height)
        new_width = width
        if resize == 'charResize':
            if (width / len_word > (charmaxW - 1)) or (width / len_word < charminW):
                if discard_wide and width / len_word > 3 * ((charmaxW - 1)):
                    return False
                if discard_narr and (width / len_word) < (charminW / 3):
                    return False
                else:
                    new_width = len_word * random.randrange(charminW, charmaxW)
        # reshape the image to the new dimensions
        im = im.resize((new_width, new_height))
        return im

def unpack_partition(new_dir, paths_list, pick=25000):
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
        os.makedirs(new_dir)
    else:
        os.makedirs(new_dir)
    c = 0
    random.shuffle(paths_list)
    for i, (path, label) in enumerate(paths_list):
        if not os.path.exists(path):
            print('%s does not exist' % path)
            return False
        try:
            im = Image.open(path)
        except:
            return False
        im = resize_im(im, label)
        if im:
            im = fix_image(im)
            im.save(os.path.join(new_dir,str(i)+'.png'))
            c += 1
            if c == pick:
                return

if __name__ == '__main__':
    dataset = 'IAM'  # CVL/IAM/RIMES/gw
    mode = 'all'  # tr/te/val/all
    # labeled = True
    top_dir = 'Datasets'
    pick = 10000
    # parameter relevant for IAM/RIMES:
    words = True  # use words images, otherwise use lines
    # parameters relevant for IAM:
    # author_number = -1  # use only images of a specific writer. If the value is -1, use all writers, otherwise use the index of this specific writer
    remove_punc = True  # remove images which include only one punctuation mark from the list ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']
    writers_images, _ = create_writers_dict(top_dir, dataset, mode, words, remove_punc)
    id = 'good'
    dir_path = os.path.join(top_dir, mode + ' '+ str(pick), id)
    paths_to_move = [labeld_image for writer_list in writers_images.values() for labeld_image in writer_list]
    unpack_partition(dir_path, paths_to_move, pick)