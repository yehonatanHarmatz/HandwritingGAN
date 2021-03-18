import os
import time

import torch

from data import create_dataset
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.util import get_curr_data
from models.StyleEncoder_model import StyleEncoder
opt = TrainOptions().parse()
print(opt)
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)
print(dataset_size)
total_iters = 0  # the total number of training iterations
opt.iter = 0
model = StyleEncoder()
visualizer = Visualizer(opt)
t_data = 0
for epoch in range(opt.epoch_count,
                   opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0
    for i, data in enumerate(dataset):
        opt.iter = i
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        total_iters += opt.batch_size * opt.num_accumulations
        epoch_iter += opt.batch_size * opt.num_accumulations
        #
        #
        #
        counter = 0
        for accumulation_index in range(opt.num_accumulations):
            curr_data = get_curr_data(data, opt.batch_size, counter)
            model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
            model.optimize()
            counter += 1
        model.optimize_step()

        if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()

        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = [model.cur_loss]
            t_comp = (time.time() - iter_start_time) / (opt.batch_size * opt.num_accumulations)
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
        '''
        if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        for i in opt.gpu_ids:
            with torch.cuda.device('cuda:%d' % (i)):
                torch.cuda.empty_cache()

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
        # model.update_learning_rate()  # update learning rates at the end of every epoch.
