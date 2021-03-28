import html
import os
from collections import defaultdict, Counter

import xmltodict
from sklearn.model_selection import train_test_split

from data.create_style_dataset import create_writers_dict


def collect_data(top_dir, dataset='IAM', words=True, remove_punc=True):
    root_dir = os.path.join(top_dir, dataset)
    output_dir = root_dir + '\\new_partition'
    writers_images = defaultdict(Counter)
    if dataset == 'IAM':
        labels_name = 'original'
        mode = ['te', 'va1', 'va2', 'tr']
        images_name = 'wordImages'

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
                    writers_images[writer].update([id.rsplit('-', 1)[0]])

    return writers_images, output_dir


def write_partition(outputPath, name, files_names):
    path = os.path.join(outputPath, name)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    with open(path, 'w') as f:
        f.writelines(files_names)


def create_partition(writers_images, outputPath, tr=0.4, val=0.4, test=0.1, gan_test=0.1):
    writers = sorted([(sum(writers_images[i].values()), int(i)) for i in writers_images.keys()], reverse=True)
    trte = float(tr + test + val).__round__(4)
    trte_writers = []
    gan_writers = []
    trte_amount = 0
    gan_amount = 0
    for amount, writer_id in writers:
        if amount < 150:
            gan_writers.append(writer_id)
            gan_amount += amount
    for amount, writer_id in writers:
        if amount < 150:
            continue
        elif trte * gan_amount < trte_amount * gan_test:
            gan_writers.append(writer_id)
            gan_amount += amount
        else:
            trte_writers.append(writer_id)
            trte_amount += amount
    print(trte_amount/gan_amount)
    gan_files = [s+'\n' for w_id in gan_writers for s in writers_images[str(w_id).zfill(3)].keys()]
    write_partition(outputPath, 'gan_test.lst', sorted(gan_files))
    # trte_files = [s+'\n' for w_id in trte_writers for s in writers_images[w_id].keys()]
    tr_files = []
    val_files = []
    te_files = []
    valtest = val+test
    for w_id in trte_writers:
        files = [f+'\n' for f in writers_images[str(w_id).zfill(3)].keys()]
        if files:
            x, y = train_test_split(files, train_size=tr/trte, test_size=(valtest)/trte)
            tr_files.extend(x)
            if len(y) > 1:
                l_val = round((val/valtest)*len(y))
                l_test = len(y) - l_val
                a, b = train_test_split(y, train_size=l_val, test_size=l_test)
                val_files.extend(a)
                te_files.extend(b)
            else:
                val_files.extend(y)
    write_partition(outputPath, 'tr.lst', sorted(tr_files))
    write_partition(outputPath, 'te.lst', sorted(te_files))
    write_partition(outputPath, 'val.lst', sorted(val_files))





if __name__ == '__main__':
    writers_images, outputPath = collect_data('Datasets', dataset='IAM')
    create_partition(writers_images, outputPath)
