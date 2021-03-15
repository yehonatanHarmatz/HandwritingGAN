import io
import os
import shutil
import random
from collections import defaultdict

import lmdb
import xmltodict
from PIL import Image
from tqdm import tqdm
import numpy as np
import html
from util.util import writeCache, dict_to_binary
from tempfile import TemporaryFile


def create_writers_dict(top_dir,dataset, mode, words, remove_punc):
    root_dir = os.path.join(top_dir, dataset)
    output_dir = root_dir + (dataset=='IAM')*('/words'*words + '/lines'*(not words))
    writers_images = defaultdict(list)
    if dataset == 'IAM':
        labels_name = 'original'
        if mode == 'all':
            mode = ['te', 'va1', 'va2', 'tr']
        elif mode == 'valtest':
            mode = ['te', 'va1', 'va2']
        else:
            mode = [mode]
        if words:
            images_name = 'wordImages'
        else:
            images_name = 'lineImages'

        images_dir = os.path.join(root_dir, images_name)
        labels_dir = os.path.join(root_dir, labels_name)
        full_ann_files = []
        im_dirs = []
        line_ann_dirs = []
        image_path_list, label_list = [], []
        for mod in mode:
            part_file = os.path.join(root_dir, 'original_partition', mod + '.lst')
            with open(part_file)as fp:
                for line in fp:
                    name = line.split('-')
                    if int(name[-1][:-1]) == 0:
                        anno_file = os.path.join(labels_dir, '-'.join(name[:2]) + '.xml')
                        full_ann_files.append(anno_file)
                        im_dir = os.path.join(images_dir, name[0], '-'.join(name[:2]))
                        im_dirs.append(im_dir)

        # if author_number >= 0:
        #     full_ann_files = [full_ann_files[author_number]]
        #     im_dirs = [im_dirs[author_number]]
        #     author_id = im_dirs[0].split('/')[-1]

        lables_to_skip = ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']
        for i, anno_file in enumerate(full_ann_files):
            with open(anno_file) as f:
                try:
                    line = f.read()
                    annotation_content = xmltodict.parse(line)
                    lines = annotation_content['form']['handwritten-part']['line']
                    writer = annotation_content['form']['@writer-id']
                    if words:
                        lines_list = []
                        for j in range(len(lines)):
                            lines_list.extend(lines[j]['word'])
                        lines = lines_list
                except:
                    print('line is not decodable')
                for line in lines:
                    try:
                        label = html.unescape(line['@text'])
                    except:
                        continue
                    if remove_punc and label in lables_to_skip:
                        continue
                    id = line['@id']
                    imagePath = os.path.join(im_dirs[i], id + '.png')
                    # image_path_list.append(imagePath)
                    # label_list.append(label)
                    writers_images[writer].append((imagePath, label))

    return writers_images, output_dir

def handle_image(imagePath, cnt, resize, imgH, h_gap, label, charmaxW, charminW, discard_wide, discard_narr,
                 init_gap, cache):
    if not os.path.exists(imagePath):
        print('%s does not exist' % imagePath)
        return False
    try:
        im = Image.open(imagePath)
    except:
        return False
    if resize in ['charResize', 'keepRatio']:
        width, height = im.size
        new_height = imgH - (h_gap * 2)
        len_word = len(label)
        width = int(width * imgH / height)
        new_width = width
        if resize == 'charResize':
            if (width / len_word > (charmaxW - 1)) or (width / len_word < charminW):
                if discard_wide and width / len_word > 3 * ((charmaxW - 1)):
                    print('%s has a width larger than max image width' % imagePath)
                    return False
                if discard_narr and (width / len_word) < (charminW / 3):
                    print('%s has a width smaller than min image width' % imagePath)
                    return False
                else:
                    new_width = len_word * random.randrange(charminW, charmaxW)

        # reshape the image to the new dimensions
        im = im.resize((new_width, new_height))
        # append with 256 to add left, upper and lower white edges
        init_w = int(random.normalvariate(init_gap, init_gap / 2))
        new_im = Image.new("RGB", (new_width + init_gap, imgH), color=(256, 256, 256))
        new_im.paste(im, (abs(init_w), h_gap))
        im = new_im

    imgByteArr = io.BytesIO()
    im.save(imgByteArr, format='tiff')
    wordBin = imgByteArr.getvalue()
    # imageKey = 'image-%02d' % cnt
    # labelKey = 'label-%09d' % cnt

    # cache[imageKey] = wordBin
    return wordBin

def create_dataset(writer_to_images_dict, outputPath, mode, k, remove_punc, resize, imgH, init_gap,
                  h_gap, charminW, charmaxW, discard_wide, discard_narr, labeled):

    outputPath = outputPath + (resize == 'charResize') * ('\\h%schar%sto%s\\' % (imgH, charminW, charmaxW)) + (
            resize == 'keepRatio') * ('\\h%s\\' % imgH) \
                 + (resize == 'noResize') * '\\noResize\\' + ('k\\' + str(k) + '\\') \
                 + mode + (resize != 'noResize') * (
                         ('_initGap%s' % init_gap) * (init_gap > 0) + ('_hGap%s' % h_gap) * (h_gap > 0)
                         + '_NoDiscard_wide' * (not discard_wide) + '_NoDiscard_wide' * (not discard_narr)) + \
                 (('IAM' in outputPath) and remove_punc) * '_removePunc'
    print(outputPath)
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)
    env = lmdb.open(outputPath, map_size=1073741824)
    cache = {}
    nSamples = 0
    cnt = 1
    for writer in tqdm(writer_to_images_dict):
        l = writer_to_images_dict[writer]
        random.shuffle(l)
        # chunks = [l[x:x + k] for x in range(0, len(l), k)]
        # if len(chunks[-1]) != k:
        #     chunks = chunks[:-1]
        # nSamples += len(chunks)
        # for chunk in chunks:
        imgs = []
        for imagePath, label in l:
            img = handle_image(imagePath, cnt, resize, imgH, h_gap, label, charmaxW, charminW, discard_wide, discard_narr,
             init_gap, cache)
            if img:
                imgs.append(img)
            if len(imgs) == k:
                style_key = 'style-%09d' % cnt
                f = io.BytesIO()
                np.save(f, imgs)
                a = f.getvalue()
                # b = np.load(io.BytesIO(a))
                cache[style_key] = a
                imgs = []
                # outfile = TemporaryFile()
                # np.save(outfile, imgs)
                # outfile.seek(0)
                # cache[style_key] = outfile.read()
                if labeled:
                    label_key = 'label-%09d' % cnt
                    cache[label_key] = str(int(writer))
                # d = {'imgs': imgs}
                # if labeled:
                #     d['label'] = str(int(writer))
                # cache[style_key] = dict_to_binary(d)
                if cnt % 150 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d' % cnt)
                cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


def main():
    dataset = 'IAM'  # CVL/IAM/RIMES/gw
    mode = 'tr'  # tr/te/val/va1/va2/all
    labeled = True
    top_dir = 'Datasets'
    # parameter relevant for IAM/RIMES:
    words = True  # use words images, otherwise use lines
    # parameters relevant for IAM:
    author_number = -1  # use only images of a specific writer. If the value is -1, use all writers, otherwise use the index of this specific writer
    remove_punc = True  # remove images which include only one punctuation mark from the list ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']

    resize = 'charResize'  # charResize|keepRatio|noResize - type of resize,
    # char - resize so that each character's width will be in a specific range (inside this range the width will be chosen randomly),
    # keepRatio - resize to a specific image height while keeping the height-width aspect-ratio the same.
    # noResize - do not resize the image
    imgH = 32  # height of the resized image
    init_gap = 0  # insert a gap before the beginning of the text with this number of pixels
    charmaxW = 17  # The maximum character width
    charminW = 16  # The minimum character width
    h_gap = 0  # Insert a gap below and above the text
    discard_wide = True  # Discard images which have a character width 3 times larger than the maximum allowed character size (instead of resizing them) - this helps discard outlier images
    discard_narr = True  # Discard images which have a character width 3 times smaller than the minimum allowed charcter size.
    k = 15  # the number of images in any unit of the dataset
    writers_images, outputPath = create_writers_dict(top_dir, dataset, mode, words, remove_punc)
    # in a previous version we also cut the white edges of the image to keep a tight rectangle around the word but it
    # seems in all the datasets we use this is already the case so I removed it. If there are problems maybe we should add this back.
    create_dataset(writers_images, outputPath, mode, k, remove_punc, resize, imgH, init_gap, h_gap,
                   charminW, charmaxW, discard_wide, discard_narr, labeled)


if __name__ == '__main__':
    main()
