import os
import time
from collections import OrderedDict

import torch
from PIL import Image
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from data import create_dataset, dataset_catalog
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.util import get_curr_data, tensor2im
from models.StyleEncoder_model import StyleEncoder

# חרמץ שלום
opt = TrainOptions().parse()
print(opt)
torch.backends.cudnn.benchmark = True
device = "cpu"
tr_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
tr_dataset_size = len(tr_dataset)
print(tr_dataset_size)
opt.dataname += "_val"
opt.dataroot = dataset_catalog.datasets[opt.dataname]
opt.scaler = GradScaler()
te_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
te_dataset_size = len(te_dataset)
if opt.batch_size_test == 0:
    opt.batch_size_test = min(te_dataset_size, opt.batch_size)
else:
    opt.batch_size_test = min(te_dataset_size, opt.batch_size_test)
print(te_dataset_size)
total_iters = 0  # the total number of training iterations
opt.iter = 0
model = StyleEncoder(opt, device=device)
visualizer = Visualizer(opt)
'''
for a in tr_dataset:
    im = tensor2im(a['style'])
    img = Image.fromarray(im, 'RGB')
    # img.save('my.png')
    img.show()
'''
t_data = 0
c_print = 0
c_save = 0
c_display = 0
for epoch in range(opt.epoch_count,
                   opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

    model.train()
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0
    # print('End of epoch %d / %d \t Time Taken: %d sec' % (
    # epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    correct = 0
    # see initial guesses
    '''
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(te_dataset):
            if i>=400:
                break
            curr_data = get_curr_data(data, opt.batch_size, 0)
            output = model(curr_data['style'])
            print(torch.max(output.data, 1)[1], data['label'])
            correct += (torch.max(output.data, 1)[1] == data['label'].to(device)).sum().item()
    accuracy = 100 * correct / te_dataset_size
    print("Test Accuracy = {}".format(accuracy))
    model.train()
    '''
    for i, data in tqdm(enumerate(tr_dataset)):
        # if i>1000/opt.batch_size:
        # break
        opt.iter = i
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters // opt.print_freq > c_print:
            t_data = iter_start_time - iter_data_time
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        #
        #
        #
        curr_data = get_curr_data(data, opt.batch_size, 0)
        model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
        model.optimize()
        model.optimize_step()
        if opt.autocast_bit:
            opt.scaler.update()
        if total_iters // opt.display_freq > c_display:
            c_display += 1
            print(model.cur_loss)
            allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            print("torch.cuda.memory_allocated: %fGB" % allocated)
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print("torch.cuda.memory_reserved: %fGB" % reserved)
            correct = 0
            # see initial guesses
            """model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, data in enumerate(te_dataset):
                    # if i >= 400:
                    #     break
                    curr_data = get_curr_data(data, opt.batch_size, 0)
                    output = model(curr_data['style'])
                    val_loss += model.loss(output, curr_data['label'].to(device))
                    print(f"Test: {torch.max(output.data, 1)[1]}, {data['label']}")
                    correct += (torch.max(output.data, 1)[1] == data['label'].to(device)).sum().item()
            accuracy = 100 * correct / te_dataset_size
            print("Test Accuracy = {}".format(accuracy))
            model.train()
            """
        # if total_iters > 1000:
        # break
        # if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
        #     save_result = total_iters % opt.update_html_freq == 0
        #     model.compute_visuals()

        if total_iters // opt.print_freq > c_print:  # print training losses and save logging information to the disk
            c_print += 1
            losses = OrderedDict()
            losses['train'] = float(model.cur_loss / epoch_iter)
            losses['val'] = float(model.val_loss / (te_dataset_size - (te_dataset_size % opt.batch_size_test)))
            print(losses)
            t_comp = (time.time() - iter_start_time) / (opt.batch_size * opt.num_accumulations)
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / tr_dataset_size, losses)

        if total_iters // opt.save_latest_freq > c_save:  # cache our latest model every <save_latest_freq> iterations
            c_save += 1
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_network(save_suffix)

        if device == 'cuda':
            for i in opt.gpu_ids:
                with torch.cuda.device('cuda:%d' % (i)):
                    torch.cuda.empty_cache()
        '''
        iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            if opt.single_writer:
                torch.save(model.z[0], os.path.join(opt.checkpoints_dir, opt.name, str(epoch) + '_z.pkl'))
                torch.save(model.z[0], os.path.join(opt.checkpoints_dir, opt.name, 'latest_z.pkl'))
        '''
    print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    correct = 0
    model.eval()
    counter_i = 0
    model.zero_loss()
    with torch.no_grad():
        for i, data in enumerate(te_dataset):
            # if i >= 200:
            # break
            counter_i += opt.batch_size_test
            curr_data = get_curr_data(data, opt.batch_size_test, 0)
            output = model(curr_data['style'], save_loss=True, train=False)
            print(f"Test: {torch.max(output.data, 1)[1]}, {data['label']}")
            correct += (torch.max(output.data, 1)[1] == data['label'].to(device)).sum().item()
    accuracy = 100 * correct / counter_i
    print("Test Accuracy = {}".format(accuracy))
    accs = OrderedDict()
    accs["Val Accuracy"] = accuracy
    correct = 0
    counter_i = 0

    with torch.no_grad():
        for i, data in enumerate(tr_dataset):
            # if i >= 200:
            # break
            counter_i += opt.batch_size
            curr_data = get_curr_data(data, opt.batch_size, 0)
            output = model(curr_data['style'])
            print(f"{i}/{tr_dataset_size // opt.batch_size}:{torch.max(output.data, 1)[1]}, {data['label']}")
            correct += (torch.max(output.data, 1)[1] == data['label'].to(device)).sum().item()
    accuracy = 100 * correct / counter_i
    print("Train Accuracy = {}".format(accuracy))
    accs["Train Accuracy"] = accuracy
    #plot the accuracies
    visualizer.plot_accuracy(epoch,1,accs)
    # model.update_learning_rate()  # update learning rates at the end of every epoch.
    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
    model.save_network(epoch)
    model.train()
