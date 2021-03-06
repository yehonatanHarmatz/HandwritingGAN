"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT

General-purpose training script for ScrabbleGAN.

You need to specify the dataset ('--dataname') and experiment name prefix ('--name_prefix').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
python train.py --name_prefix demo --dataname RIMEScharH32W16 --capitalize --display_port 8192

See options/base_options.py and options/train_options.py for more training options.
"""
import time

from torch.cuda.amp import GradScaler

from options.train_options import TrainOptions
from data import create_dataset, dataset_catalog
from models import create_model
from util.visualizer import Visualizer
from util.util import seed_rng
from util.util import prepare_z_y, get_curr_data
import torch
import os

if __name__ == '__main__':
    torch.set_num_threads(1)
    opt = TrainOptions().parse()   # get training options
    # Seed RNG
    seed_rng(opt.seed)
    torch.backends.cudnn.benchmark = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    #prev_name=opt.dataname
    #prev_mode=opt.dataset_mode
    #change the name of dataset
    opt_style= TrainOptions().parse()
    opt_style.dataname='style15IAMcharH32rmPunct_all'
    opt_style.dataroot = dataset_catalog.datasets[opt_style.dataname]
    opt_style.dataset_mode ="style"
    #opt_style.batch_size=opt_style.style_batch_size
    print(opt_style.batch_size)
    tr_dataset_style = create_dataset(opt_style)  # create a dataset given opt.dataset_mode and other options
    tr_dataset_size = len(tr_dataset_style)
    print('The number of  Style training images = %d' % tr_dataset_size)
    #opt.style_opt = opt_style
    #opt.dataname=prev_name
    #opt.dataset_mode=prev_mode
    #TODO- handle 16bit in GAN and Dw
    # OUR VARIABLE
    if opt.autocast_bit:
        opt.scaler = GradScaler()
    model = create_model(opt,opt_style)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.single_writer:
        opt.G_init='N02'
        opt.D_init='N02'
        model.netG.init_weights()
        model.netD.init_weights()
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    total_iters_style = 0  # the total number of training iterations
    opt.iter = 0
    counter_print = 0
    counter_save = 0
    counter_display = 0
    # seed_rng(opt.seed)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter_style = 0

        # enumerate the 2 datasets toegther
        style_iterator = iter(tr_dataset_style)

        for i, data in enumerate(dataset):

            try:
                data_style = next(style_iterator)
            except StopIteration:
                style_iterator = iter(tr_dataset_style)
                data_style = next(style_iterator)

        #do_cool_things()
        #for i, data in enumerate(dataset):  # inner loop within one epoch
            opt.iter = i
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size*opt.num_accumulations
            epoch_iter += opt.batch_size*opt.num_accumulations
            total_iters_style += opt_style.batch_size * opt.num_accumulations
            epoch_iter_style += opt_style.batch_size * opt.num_accumulations
            # default = 4
            if opt.num_critic_train == 1:
                counter = 0
                for accumulation_index in range(opt.num_accumulations):
                    curr_data = get_curr_data(data, opt.batch_size, counter)
                    curr_data_style = get_curr_data(data_style, opt_style.batch_size, 0)
                    model.set_input(curr_data,style_img=curr_data_style)  # unpack data from dataset and apply preprocessing
                    model.optimize_G()
                    model.optimize_D_OCR()
                    counter += 1
                model.optimize_G_step()
                model.optimize_D_OCR_step()
                if opt.autocast_bit:
                    opt.scaler.update()
            # defulat=4 so else
            else:
                if (i % opt.num_critic_train) == 0:
                    counter = 0
                    # defult=1
                    for accumulation_index in range(opt.num_accumulations):
                        #TODO- add the data from style
                        curr_data = get_curr_data(data, opt.batch_size, counter)
                        curr_data_style = get_curr_data(data_style, opt_style.batch_size, 0)
                        model.set_input(curr_data,style_img=curr_data_style)   # unpack data from dataset and apply preprocessing
                        model.optimize_G()
                        counter += 1
                    model.optimize_G_step()
                counter = 0
                for accumulation_index in range(opt.num_accumulations):
                    curr_data = get_curr_data(data, opt.batch_size, counter)
                    curr_data_style = get_curr_data(data_style, opt_style.batch_size, 0)
                    model.set_input(curr_data,
                                    style_img=curr_data_style)  # unpack data from dataset and apply preprocessing
                    #model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                    model.optimize_D_OCR()
                    counter += 1
                model.optimize_D_OCR_step()
                if opt.autocast_bit:
                    opt.scaler.update()
                # print(model.netG.linear.bias[:10])
                # print('G',model.loss_G, 'D', model.loss_D, 'Dreal',model.loss_Dreal, 'Dfake', model.loss_Dfake,
                #       'OCR_real', model.loss_OCR_real, 'OCR_fake', model.loss_OCR_fake, 'grad_fake_OCR', model.loss_grad_fake_OCR, 'grad_fake_adv', model.loss_grad_fake_adv)



            if total_iters % opt.display_freq == 0 or total_iters // opt.display_freq> counter_display:   # display images on visdom and save images to a HTML file
                counter_display += 1
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(),model.get_current_fake_labels(),epoch, save_result)
                #TODO- add
                visualizer.plot_current_style(*model.get_current_style())

            if total_iters % opt.print_freq == 0 or total_iters // opt.print_freq> counter_print:    # print training losses and save logging information to the disk
                counter_print += 1
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / (opt.batch_size*opt.num_accumulations)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                print("torch.cuda.memory_allocated: %fGB" % allocated)
                reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
                print("torch.cuda.memory_reserved: %fGB" % reserved)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0 or total_iters // opt.save_latest_freq> counter_save:   # cache our latest model every <save_latest_freq> iterations
                counter_save += 1
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            for i in opt.gpu_ids:
                with torch.cuda.device('cuda:%d' % (i)):
                    torch.cuda.empty_cache()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            if opt.single_writer:
                torch.save(model.z[0], os.path.join(opt.checkpoints_dir, opt.name, str(epoch)+'_z.pkl'))
                torch.save(model.z[0], os.path.join(opt.checkpoints_dir, opt.name, 'latest_z.pkl'))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
