from data.style_dataset import StyleDataset
from models.OCR_network import strLabelConverter
from models.StyleEncoder_model import StyleEncoder
from options.train_options import TrainOptions
from util.util import prepare_z_y
from util.visualizer import Visualizer


class Generator:
    def __init__(self, g_model, s_extractor, words_encoder, z_dim, visualiser=None):
        self.g_model = g_model
        self.g_model.eval()
        self.s_extractor = s_extractor
        self.s_extractor.eval()
        self.words_encoder = words_encoder
        self.words_encoder.eval()
        self.vis = visualiser
        self.z_dim = z_dim

    def generate_word_image(self, style, word):
        style_tensor = style['style']
        writer = style['label']
        style_features = self.s_extractor(style_tensor)
        word_encode = self.words_encoder.encode(word.encode('utf-8'))
        z, _ = prepare_z_y(1, self.z_dim, 1)
        result_tensor = self.g_model(word_encode, z, style_features)
        if self.vis:
            self.vis.plot_result_style(result_tensor, style_tensor, word, writer)


# TODO - Complete
opt = TrainOptions().parse()
g = #load_G
path_s="C:\\Users\\Ron\\PycharmProjects\\HandwritingGANgit\checkpoints\\demo_autocast_debug_style15IAMcharH32rmPunct_GANres16_bs128\\bast_accuracy_val81.640625_net_Style_Encoder.pth"
s = StyleEncoder(opt, already_trained=True, features_only=True, path=path_s).to(opt.device)
w = strLabelConverter(opt.alphabet)
vis = Visualizer(opt)
gen = Generator(g, s, w, opt.dim_z, vis)
opt_style_test = TrainOptions().parse()
opt_style_test.blabla
opt_style_test.blabla
opt_style_test.blabla
style_test_dataset = StyleDataset(opt_style_test)
for i in range(len(style_test_dataset)):
    gen.generate_word_image(style_test_dataset[i], some_word)