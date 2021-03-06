import ast
import os
import random
import shutil
from time import sleep
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from data import dataset_catalog
from data.organize_data import fix_image
from data.style_dataset import StyleDataset
from data.text_dataset import TextDataset
from models.BigGAN_networks import Generator
from models.OCR_network import strLabelConverter
from models.StyleEncoder_model import StyleEncoder
from options.train_options import TrainOptions
from util.util import prepare_z_y, make_one_hot, concat_images, tensor2im
from util.visualizer import Visualizer

class OurGenerator:
    def __init__(self, g_model, s_extractor, words_encoder, z_dim, opt, visualiser=None):
        self.g_model = g_model
        self.g_model.eval()
        self.s_extractor = s_extractor
        self.s_extractor.eval()
        self.words_encoder = words_encoder
        self.vis = visualiser
        self.z_dim = z_dim
        self.opt = opt
        opt.n_classes = len(opt.alphabet)
        exception_chars = ['ï', 'ü', '.', '_', 'ö', ',', 'ã', 'ñ']

        if opt.lex.endswith('.tsv'):
            self.lex = pd.read_csv(opt.lex, sep='\t')['lemme']
            self.lex = [word.split()[-1] for word in self.lex if
                        (pd.notnull(word) and all(char not in word for char in exception_chars))]
        elif opt.lex.endswith('.txt'):
            with open(opt.lex, 'rb') as f:
                self.lex = f.read().splitlines()
            lex=[]
            for word in self.lex:
                try:
                    word=word.decode("utf-8")
                except:
                    continue
                if len(word)<20:
                    lex.append(word)
            self.lex = lex
        else:
            raise ValueError('could not load lexicon ')

    def generate_word_image(self, style, word):
        style_tensor = style['style']
        writer = style['label']
        style_features = self.s_extractor(style_tensor.unsqueeze(0))
        # word_encode = self.words_encoder.encode(word.encode('utf-8'))
        text_encode_fake, len_text_fake = self.words_encoder.encode([word.encode('utf-8')])
        # convert to device
        text_encode_fake = text_encode_fake.to(self.opt.device)
        z, _ = prepare_z_y(1, self.z_dim, 1, device=self.opt.device)
        z.sample_()
        if self.opt.one_hot:
            one_hot_fake = make_one_hot(text_encode_fake, len_text_fake, self.opt.n_classes).to(
                self.opt.device)
            result_tensor = self.g_model(z, one_hot_fake, style_features.squeeze())
            ones_img = torch.ones(result_tensor.shape, dtype=torch.float32)
            ones_img[:, :, :, 0:result_tensor.shape[3]] = result_tensor[0, :, :, 0:ones_img.shape[3]]
            result_tensor = ones_img
        # result_tensor = self.g_model(z, word_encode, style_features)
        #     if self.vis:
        #         self.vis.plot_result_style(result_tensor, style['original'].unsqueeze(0), word, writer)
            return result_tensor

    def plot_result(self, result, original, word, writer):
        if self.vis:
            self.vis.plot_result_style(result, original, word, writer)

    def plot_image(self, image, title):
        if self.vis:
            self.vis.plot_image(image, title)

    def gen_one_batch(self, path_to_save, style_dataset, bs, c, for_fid):
        style_len = len(style_dataset)
        z, y = prepare_z_y(bs, self.z_dim, len(self.lex), device=self.opt.device, seed=random.randint(0, 20000000))
        z.sample_()
        y.sample_()
        words = [self.lex[int(i)].encode('utf-8') for i in y]
        style = [style_dataset[k] for k in random.choices(range(style_len), k=bs)]
        style_tensor = torch.cat([s['style'].unsqueeze(0) for s in style],dim=0).to(self.opt.device)
        #tensors, dim=0, *, out=None)
        style_features = self.s_extractor(style_tensor)
        # word_encode = self.words_encoder.encode(word.encode('utf-8'))
        text_encode_fake, len_text_fake = self.words_encoder.encode(words)
        # convert to device
        text_encode_fake = text_encode_fake.to(self.opt.device)
        if self.opt.one_hot:
            one_hot_fake = make_one_hot(text_encode_fake, len_text_fake, self.opt.n_classes).to(self.opt.device)
            result_tensor = self.g_model(z, one_hot_fake, style_features.squeeze().to(self.opt.device))
            ones_img = torch.ones(result_tensor.shape, dtype=torch.float32)
            for res in range(bs):
                ones_img[:, :, :, 0:result_tensor.shape[3]] = result_tensor[res, :, :, 0:ones_img.shape[3]]
                # r = ones_img.squeeze()
                r = ones_img
                img_pil = Image.fromarray(tensor2im(r), 'RGB')#.convert('L')
                # img_pil.show()
                if for_fid:
                    im = fix_image(img_pil)
                else:
                    im = img_pil
                im.save(os.path.join(path_to_save, str(c) + '.png'))
                c += 1
        return c

    def generate_and_save(self, path_to_save, amount, style_dataset, bs=1, add=True, for_fid=True):
        if os.path.exists(path_to_save):
            if not add:
                shutil.rmtree(path_to_save)
                os.makedirs(path_to_save)
        else:
            os.makedirs(path_to_save)
        # style_len = len(style_dataset)
        c = 0
        for i in tqdm(range(amount//bs)):
            c = self.gen_one_batch(path_to_save, style_dataset, bs, c, for_fid)
        new_bs = amount - c
        if new_bs > 0:
            c = self.gen_one_batch(path_to_save, style_dataset, new_bs, c, for_fid)
        print('generated and saved ' + str(c) + 'images')
                    # result_tensor = ones_img


def rotate(l, n):
    return l[-n:] + l[:-n]


def load_g(path, opt):
    opt.n_classes = len(opt.alphabet)
    g = Generator(**vars(opt))
    state_dict = torch.load(path, map_location=str(opt.device))
    # g = torch.load(path, map_location=str('cpu'))
    # return g
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        # self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        keys = key.split('.')
        if keys[0] == 'module': keys = keys[1:]
    g.load_state_dict(state_dict)
    g=g.to(opt.device)
    print(g)
    return g

def gen_multi_style(gen, word, style_list, s_list_1):
    # res = [gen.generate_word_image(s, word).squeeze(1) for s in style_list]
    res = [generate_single_sentence(gen, word.split(), s).squeeze(1) for s in style_list]
    res = concat_images(res, result_h=(sum(res1.shape[1] for res1 in res)))
    s_res = [s_1['original'] for s_1 in s_list_1]
    s_res = concat_images(s_res, result_h=(sum(res1.shape[1] for res1 in s_res)))
    width_list = [s_res.shape[2] + 15]
    width_list.append(res.shape[2])
    image = concat_images([s_res, res], result_h=max(res1.shape[1] for res1 in res),result_w=(res.shape[2]+s_res.shape[2]), width_list=width_list, sentence=False, dim=2)
    gen.plot_image(image.unsqueeze(0), 'indexes: ' +str([s['idx'] for s in style_list]))

def generate_single_sentence(gen, words, style):
    res = [gen.generate_word_image(style, word).squeeze(1)
           for word in words]
    width_list = [word.shape[2] + 15 for word in res][:-1]
    width_list.append(res[-1].shape[2])
    res_tensor = concat_images(res, result_h=max(res1.shape[1] for res1 in res), width_list=width_list, sentence=True, dim=2)
    return res_tensor

def generate_sentence(gen, style_test_dataset, words):
    #for i in range(0, len(style_test_dataset), 1):

    #word = random.choice(words_test_dataset)['label'].decode('utf-8')
    for i in range(0,len(style_test_dataset),10):
        res = [gen.generate_word_image(style_test_dataset[i], word).squeeze(1)
               for word in words]
        width_list = [word.shape[2]+15 for word in res][:-1]
        width_list.append(res[-1].shape[2])
        res_tensor = concat_images(res, result_h=max(res1.shape[1] for res1 in res),width_list=width_list, sentence=True, dim=2)
        gen.plot_image(res_tensor.unsqueeze(0),str(words)+' '+str(style_test_dataset[i]['label']))
        sleep(0.5)

def generate_sentences(gen, style_test_dataset, words):
    #for i in range(0, len(style_test_dataset), 1):
    sentences = ['']
    i = 0
    for word in words:
        if len(sentences[i]) + len(word) + 1 < 60:
            sentences[i] += f' {word}'
        else:
            sentences[i] = sentences[i].split()
            i += 1
            sentences.append(f'{word}')
    if i < len(sentences):
        sentences[i] = sentences[i].split()
    for i in range(0,len(style_test_dataset),10):
        imgs = [generate_single_sentence(gen, sentence, style_test_dataset[i]) for sentence in sentences]
        res_tensor = concat_images(imgs, result_h=sum(img.shape[1] for img in imgs), result_w=max(img.shape[2] for img in imgs), dim=1)
        # gen.plot_image(res_tensor.unsqueeze(0),str(words[:4])+' '+str(style_test_dataset[i]['label']))
        # sleep(0.2)
        gen.plot_result(res_tensor.unsqueeze(0), style_test_dataset[i]['original'].unsqueeze(0), 'a', style_test_dataset[i]['label'])


def generate_task_3(gen, style_test_dataset, words_test_dataset, style_test_dataset_3):
    cur = 0
    min_i = 0
    max_i = 0
    words = []
    for i in range(0, len(style_test_dataset), 1):
        # if int(style_test_dataset[i]['label']):
        if int(style_test_dataset[i]['label']) == cur:
            max_i += 1
        else:
            for d in range(max_i - min_i + 3):
                imgs = []
                for k in range(5):
                    words = [random.choice(words_test_dataset)['label'].decode('utf-8') for m in range(1)]
                    res = [gen.generate_word_image(style_test_dataset[random.randint(min_i, max_i)], word).squeeze(1)
                           for word in words]
                    res_tensor = concat_images(res, result_h=sum(res1.shape[1] for res1 in res))
                    org_3 = style_test_dataset_3[random.randint(min_i, max_i)]['original']
                    org_2b = style_test_dataset_3[random.randint(min_i, max_i)]['original']
                    lt = [org_3, res_tensor, org_2b]
                    lt = rotate(lt, k - 3)
                    # c = list(zip(lt, ['X','O','O']))
                    # random.shuffle(c)
                    # lt, flag = zip(*c)
                    imgs.append(concat_images(lt, result_h=(res_tensor.shape[1] + org_3.shape[1]), dim=1))
                img = concat_images(imgs, result_h=max(imgs[h].shape[1] for h in range(len(imgs))), dim=2)
                # gen.plot_result(res, style_test_dataset_3[i]['original'].unsqueeze(0), word, style_test_dataset[i]['label'])
                gen.plot_image(img.unsqueeze(0),
                               f'BIG, style {style_test_dataset[max_i]["label"]}, words {str(words)[1:-1]}')
                sleep(0.1)
            min_i = i
            max_i = i
            cur = int(style_test_dataset[i]['label'])

def generate_task_1_comp(gen, style_test_dataset, words_test_dataset, word=None):
    const = word is not None
    for i in range(0, len(style_test_dataset), 1):
        if not const:
            word = random.choice(words_test_dataset)['label'].decode('utf-8')
        res = gen.generate_word_image(style_test_dataset[i], word)
        gen.plot_result(res, style_test_dataset[i]['original'].unsqueeze(0), word, style_test_dataset[i]['label'])
        sleep(0.1)

def search_writers(style_dataset, writers):
    from collections import defaultdict
    d = defaultdict(list)
    for i in tqdm(range(len(style_dataset))):
        s = style_dataset[i]
        if s['label'] in writers:
            d[s['label']].append(i)
    return d

def main():
    torch.set_num_threads(1)
    print("harmatz lets finish this already ssksnm,fgx,mbvcvbcvbtguiu")
    opt = TrainOptions().parse()
    opt.device = 'cpu'
    g = load_g('checkpoints/final_models/10_net_G.pth', opt)
    # g = load_g(r'C:\Users\Ron\PycharmProjects\HandwritingGANgit\checkpoints\demo_autocast_final3cont_IAMcharH32rmPunct_all_CapitalizeLex_GANres16_bs16_mixed_precs\4_net_G.pth', opt)
    # path_s =r"C:\Users\Ron\PycharmProjects\HandwritingGANgit\checkpoints\\demo_paper_resnet18_steplr_style15IAMcharH32rmPunct_GANres16_bs128\bast_accuracy_val94.84375_net_Style_Encoder.pth"

    path_s = ".\\checkpoints\\final_models\\bast_accuracy_val94.84375_net_Style_Encoder.pth"
    # g=load_g(r'C:\Users\Ron\PycharmProjects\HandwritingGANgit\checkpoints\demo_autocast_final_IAMcharH32rmPunct_all_CapitalizeLex_GANres16_bs16_mixed_precs\latest_net_G-old.pth', opt)
    # path_s="C:\\Users\\Ron\\PycharmProjects\\HandwritingGANgit\checkpoints\\demo_autocast_debug_style15IAMcharH32rmPunct_GANres16_bs128\\bast_accuracy_val81.640625_net_Style_Encoder.pth"
    s = StyleEncoder(opt, already_trained=True, features_only=True, path=path_s, device=opt.device).to(opt.device)
    w = strLabelConverter(opt.alphabet)
    vis = None
    vis = Visualizer(opt)
    gen = OurGenerator(g, s, w, opt.dim_z, opt, vis)

    opt_style_test = TrainOptions().parse()
    opt_style_test.dataname = 'style15IAMcharH32rmPunct_gan'
    opt_style_test.dataroot = dataset_catalog.datasets[opt_style_test.dataname]
    opt_style_test.test = True
    opt_style_test.device = opt.device
    style_test_dataset = StyleDataset(opt_style_test)

    opt_style_test_1 = TrainOptions().parse()
    opt_style_test_1.k = 1
    opt_style_test_1.dataname = 'style15IAMcharH32rmPunct_gan'
    opt_style_test_1.dataroot = dataset_catalog.datasets[opt_style_test.dataname]
    opt_style_test_1.test = True
    style_test_dataset_1 = StyleDataset(opt_style_test_1)
    indexes = [241, 7, 1594, 1610, 180, 68, 365, 269, 1460, 770, 794, 1181, 1041, 337]
    indexes = [337, 241, 180, 68, 180, 7, 365, 1041, 1610, 1594, 1460]
    while True:
        s_list = [style_test_dataset[i] for i in indexes]
        s_list_1 = [style_test_dataset_1[i] for i in indexes]
        word = 'Example of the same sentence with different styles'  # random.choice(['Israel','bar-ilan-university'])
        gen_multi_style(gen, word, s_list, s_list_1)
    '''
    good_writers = [26, 84, 188, 210, 237, 240, 243, 244, 261, 273]
    d = search_writers(style_test_dataset, good_writers)
    while True:
        indexes = [random.choice(d[writer]) for writer in good_writers]
        s_list = [style_test_dataset[i] for i in indexes]
        s_list_1 = [style_test_dataset_1[i] for i in indexes]
        word = 'bar-ilan-university'#random.choice(['Israel','bar-ilan-university'])
        gen_multi_style(gen, word, s_list, s_list_1)

    '''
    l = list(range(len(style_test_dataset)))
    while True:
        indexes = random.sample(l, 15)
        s_list = []
        s_list_1 = []
        for i in indexes:
            s = style_test_dataset[i]
            if s['label'] not in s_list:
                s_list.append(s)
                s_list_1.append(style_test_dataset_1[i])
                if len(s_list) == 10:
                    gen_multi_style(gen, 'Example of the same sentence with different styles', s_list, s_list_1)
                    break

    '''
    generate_sentences(gen, style_test_dataset, 'Alonso et al first presented a solution using Conditional GAN \
                                                -conditioned on wanted word- to generate cursive text images of given \
                                                words but no specified style for the image \
                                                The network presented in Alonso work uses LSTM for encoding input word conditional \
                                                Generator and Discriminator based on BigGAN network and R - RNN-CNN - recognizer for OCR \
                                                while R is trained on real samples only with CTC loss and D uses adversarial loss the loss \
                                                of G is the sum of D and R on the fakes generated'.split())
    '''
    # gen.generate_and_save(".\gan_forward_new2",50,style_test_dataset, bs=8)
    exit(1)

    opt = TrainOptions().parse()
    opt.device = 'cpu'
    # g = load_g('.\\checkpoints\\final_models\\latest_net_G-old.pth', opt)
    # path_s = ".\\checkpoints\\final_models\\bast_accuracy_val81.640625_net_Style_Encoder.pth"
    # s = StyleEncoder(opt, already_trained=True, features_only=True, path=path_s, device=opt.device).to(opt.device)
    # w = strLabelConverter(opt.alphabet)
    # vis = Visualizer(opt)
    # gen = OurGenerator(g, s, w, opt.dim_z, opt, vis)

    opt_style_test = TrainOptions().parse()
    opt_style_test.dataname = 'style15IAMcharH32rmPunct_gan'
    opt_style_test.dataroot = dataset_catalog.datasets[opt_style_test.dataname]
    opt_style_test.test = True
    style_test_dataset = StyleDataset(opt_style_test)

    opt_words = TrainOptions().parse()
    opt_words.dataname = 'IAMcharH32rmPunct_gan'
    opt_words.dataroot = dataset_catalog.datasets[opt_words.dataname]
    opt_words.test = True
    opt_words.device = 'cpu'
    words_test_dataset = TextDataset(opt_words)
    l = list(range(0, len(words_test_dataset), 1))
    # random.shuffle(l)

    generate_task_1_comp(gen, style_test_dataset, words_test_dataset, word='yehonatan-harmatz-was-here')
    '''
    opt_style_test_3 = TrainOptions().parse()
    opt_style_test_3.k = 3
    opt_style_test_3.dataname = 'style15IAMcharH32rmPunct_gan'
    opt_style_test_3.dataroot = dataset_catalog.datasets[opt_style_test.dataname]
    opt_style_test_3.test = True
    style_test_dataset_3 = StyleDataset(opt_style_test_3)

    
    same = []
    w_same = None
    '''
    '''
    2 vs 1
    for i in range(0, len(style_test_dataset), 1):
        # if int(style_test_dataset[i]['label']):
        words = ['harmatz', 'hi', 'israel'] #ast.literal_eval(style_test_dataset[i]['words'])[0]
        words = [random.choice(words_test_dataset)['label'].decode('utf-8') for m in range(1)]
        res = [gen.generate_word_image(style_test_dataset[i], word).squeeze(1) for word in words]
        res_tensor = concat_images(res, result_h=sum(res1.shape[1] for res1 in res))
        org_3 = style_test_dataset_3[i]['original']
        org_2b = style_test_dataset_3[i]['original']
        lt =[res_tensor, org_3, org_2b]
        c = list(zip(lt, ['X','O','O']))
        random.shuffle(c)
        lt, flag = zip(*c)
        img = concat_images(lt, result_h=(res_tensor.shape[1] + org_3.shape[1]), dim=1)
        # gen.plot_result(res, style_test_dataset_3[i]['original'].unsqueeze(0), word, style_test_dataset[i]['label'])
        gen.plot_image(img.unsqueeze(0), f'ORD {str(flag)}, style {style_test_dataset[i]["label"]}, words {str(words)[1:-1]}')
        sleep(0.1)
    '''

    '''

    for i in l:
        # if int(style_test_dataset[i]['label']):
        # gen.generate_word_image(style_test_dataset[i], ast.literal_eval(style_test_dataset[i]['words'])[0])
        w = words_test_dataset[i]['writer']
        if len(words_test_dataset[i]['label'].decode()) > 2:
            # vis.plot_word(words_test_dataset[i]['img'].unsqueeze(0), words_test_dataset[i]['label'])
            # sleep(1)
            if w == w_same or not w_same:
                w_same = w
                same.append(words_test_dataset[i])
                if len(same) == 3:
                    img = concat_images([same_i['img'] for same_i in same], result_h=sum(res1["img"].shape[1] for res1 in same))
                    words = [word["label"].decode() for word in same]
                    gen.plot_image(img.unsqueeze(0), f'SAME, style {w}, words {str(words)[1:-1]}')
                    w_same = None
                    same = []
                    sleep(0.1)
            elif w > w_same:
                w_same = w
                same = []
                same.append(words_test_dataset[i])
    """
    '''
    '''
    # random.shuffle(l)
    diff = []
    w_diff = []
    for i in l:
        w = words_test_dataset[i]['writer']
        if (w in w_diff and len(w_diff) < 2) or (w not in w_diff and len(w_diff) != 1):
            diff.append(words_test_dataset[i])
            w_diff.append(w)
            if len(diff) == 3:
                img = concat_images([same_i['img'] for same_i in diff], result_h=sum(res1["img"].shape[1] for res1 in same))
                words = [word["label"].decode() for word in diff]
                gen.plot_image(img.unsqueeze(0), f'DIFF, style {str(w_diff)[1:-1]}, words {str(words)[1:-1]}')
                w_diff = []
                diff = []
                sleep(0.1)
        elif len(w_diff) == 1:
            diff = [words_test_dataset[i]]
            w_diff = [w]
        elif len(w_diff) == 2:
            j = random.choice(l)
            w = words_test_dataset[j]['writer']
            if (w in w_diff and len(w_diff) < 2) or (w not in w_diff and len(w_diff) != 1):
                diff.append(words_test_dataset[j])
                w_diff.append(w)
                if len(diff) == 3:
                    random.shuffle(diff)
                    img = concat_images([same_i['img'] for same_i in diff],
                                        result_h=sum(res1["img"].shape[1] for res1 in same))
                    words = [word["label"].decode() for word in diff]
                    gen.plot_image(img.unsqueeze(0), f'DIFF, style {str(w_diff)[1:-1]}, words {str(words)[1:-1]}')
                    w_diff = []
                    diff = []
                    sleep(0.1)
    # '''



if __name__ == '__main__':
    main()