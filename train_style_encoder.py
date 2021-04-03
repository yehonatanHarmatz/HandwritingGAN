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

def main():
    torch.set_num_threads(1)
    opt_tr = TrainOptions().parse()
    print(opt_tr)
    torch.backends.cudnn.benchmark = True
    device = opt_tr.device
    tr_dataset = create_dataset(opt_tr)  # create a dataset given opt.dataset_mode and other options
    tr_dataset_size = len(tr_dataset)
    print(tr_dataset_size)
    opt_val = TrainOptions().parse()
    opt_val.dataname += "_val"
    opt_val.dataroot = dataset_catalog.datasets[opt_val.dataname]
    if opt_tr.autocast_bit:
        opt_tr.scaler = GradScaler()
    #opt_val.scaler = GradScaler()
    opt_val.test=True
    te_dataset = create_dataset(opt_val)  # create a dataset given opt.dataset_mode and other options
    te_dataset_size = len(te_dataset)
    if opt_val.batch_size_test == 0:
        #opt_tr.batch_size_test = min(te_dataset_size, opt_tr.batch_size)
        opt_val.batch_size_test = min(te_dataset_size, opt_val.batch_size)
    else:
        opt_val.batch_size_test = min(te_dataset_size, opt_val.batch_size_test)
    print(te_dataset_size)
    total_iters = 0  # the total number of training iterations
    opt_tr.iter = 0
    model = StyleEncoder(opt_tr, device=device)
    visualizer = Visualizer(opt_tr)
    best_val_acc = 0
    best_val_loss = float('inf')

    # TODO PUT this code where needed
    '''
    Look here Sherman 
    from data.style_dataset import StyleDataset
    style_dataset = StyleDataset(opt_tr)
    style_zero = style_dataset[0]
    style_zero_features = model(style_zero['style'].unsqueeze(0))
    '''
    '''
    for a in tr_dataset:
        im = tensor2im(a['img'])
        visualizer.plot_current_style(im, str(int(a['writer'][0].decode())))
        time.sleep(1)
        # img = Image.fromarray(im, 'RGB')
        # # img.save('my.png')
        # img.show()
    # '''
    t_data = 0
    c_print = 0
    c_save = 0
    c_display = 0
    first = True
    for epoch in range(opt_tr.epoch_count,
                       opt_tr.niter + opt_tr.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

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
            opt_tr.iter = i
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters // opt_tr.print_freq > c_print:
                t_data = iter_start_time - iter_data_time
            total_iters += opt_tr.batch_size
            epoch_iter += opt_tr.batch_size
            #
            #
            #
            curr_data = get_curr_data(data, opt_tr.batch_size, 0)
            model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
            model.optimize()
            model.optimize_step()
            if opt_tr.autocast_bit:
                opt_tr.scaler.update()
            if total_iters // opt_tr.display_freq > c_display:
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

            if total_iters // opt_tr.print_freq > c_print:  # print training losses and save logging information to the disk
                c_print += 1
                losses = OrderedDict()
                losses['train'] = float(model.cur_loss / epoch_iter)
                if first:
                    correct = 0
                    model.eval()
                    counter_i = 0
                    correct_per_writer = [(0, 0)] * 140
                    model.zero_loss()
                    with torch.no_grad():
                        for i, data in enumerate(te_dataset):
                            # if i >= 200:
                            # break
                            counter_i += opt_val.batch_size_test
                            curr_data = get_curr_data(data, opt_val.batch_size_test, 0)
                            output = model(curr_data['style'], save_loss=True, train=False)
                            print(f"Test {i}/{te_dataset_size // opt_val.batch_size_test}: {torch.max(output.data, 1)[1]}, {data['label']}")
                            for w_pred, w_real in zip(torch.max(output.data, 1)[1], data['label']):
                                w_p_int = int(w_pred.decode())
                                w_r_int = int(w_real.decode())
                                _, y = correct_per_writer[w_p_int]
                                x, _ = correct_per_writer[w_r_int]
                                if w_p_int == w_r_int:
                                    x += 1
                                correct_per_writer[w_r_int] = x, y + 1
                            correct += (torch.max(output.data, 1)[1] == data['label'].to(device)).sum().item()
                        accuracy = 100 * correct / counter_i
                        micro, macro = calc_precision(correct_per_writer)
                        best_val_acc, best_val_loss = save_best_model(model, best_val_acc, best_val_loss, accuracy, te_dataset_size, opt_val)
                    first = False
                losses['val'] = float(model.val_loss / (te_dataset_size - (te_dataset_size % opt_val.batch_size_test)))
                print(losses)
                t_comp = (time.time() - iter_start_time) / (opt_tr.batch_size * opt_tr.num_accumulations)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.print_precision(epoch, epoch_iter, {'micro': micro, 'macro': macro}, t_comp, t_data)
                if opt_tr.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / tr_dataset_size, losses)

            if total_iters // opt_tr.save_latest_freq > c_save:  # cache our latest model every <save_latest_freq> iterations
                c_save += 1
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt_tr.save_by_iter else 'latest'
                model.save_network(save_suffix)

            if device == 'cuda':
                for i in opt_tr.gpu_ids:
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
            epoch, opt_tr.niter + opt_tr.niter_decay, time.time() - epoch_start_time))
        correct = 0
        correct_per_writer = [(0, 0)] * 140
        model.eval()
        counter_i = 0
        model.zero_loss()
        with torch.no_grad():
            for i, data in enumerate(te_dataset):
                # if i >= 200:
                # break
                counter_i += opt_val.batch_size_test
                curr_data = get_curr_data(data, opt_val.batch_size_test, 0)
                output = model(curr_data['style'], save_loss=True, train=False)
                print(f"Test {i}/{te_dataset_size // opt_val.batch_size_test}: {torch.max(output.data, 1)[1]}, {data['label']}")
                for w_pred, w_real in zip(torch.max(output.data, 1)[1], data['label']):
                    w_p_int = int(w_pred.decode())
                    w_r_int = int(w_real.decode())
                    _, y = correct_per_writer[w_p_int]
                    x, _ = correct_per_writer[w_r_int]
                    if w_p_int == w_r_int:
                        x += 1
                    correct_per_writer[w_r_int] = x, y + 1
                correct += (torch.max(output.data, 1)[1] == data['label'].to(device)).sum().item()
        micro, macro = calc_precision(correct_per_writer)
        accuracy = 100 * correct / counter_i
        print("Test Accuracy = {}".format(accuracy))
        accs = OrderedDict()
        accs["Val Accuracy"] = accuracy
        first = False
        best_val_acc, best_val_loss = save_best_model(model, best_val_acc, best_val_loss, accuracy, te_dataset_size, opt_val)
        correct = 0
        counter_i = 0
        with torch.no_grad():
            for i, data in enumerate(tr_dataset):
                # if i >= 200:
                # break
                counter_i += opt_tr.batch_size
                curr_data = get_curr_data(data, opt_tr.batch_size, 0)
                output = model(curr_data['style'])
                print(f"{i}/{tr_dataset_size // opt_tr.batch_size}:{torch.max(output.data, 1)[1]}, {data['label']}")
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


def save_best_model(model, best_val_acc, best_val_loss, accuracy, te_dataset_size, opt):
    if accuracy > best_val_acc:
        model.save_network('bast_accuracy_val'+str(accuracy))
        best_val_acc = accuracy
    loss = float(model.val_loss / (te_dataset_size - (te_dataset_size % opt.batch_size_test)))
    if loss < best_val_loss:
        model.save_network('bast_loss_val'+str(loss))
        best_val_loss = loss
    return best_val_acc, best_val_loss


def calc_precision(data_list):
    precision_list = [a/b for a, b in data_list]
    import numpy as np
    macro = np.avarge(precision_list)
    micro = np.sum([a for a, _ in data_list])/np.sum([a for _, a in data_list])
    return micro, macro

if __name__ == '__main__':
    main()

