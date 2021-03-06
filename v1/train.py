from __future__ import print_function
from __future__ import division
import os
import argparse
import torch
import pathlib
import time
import numpy as np
from tqdm.autonotebook import tqdm
from model import Wavenet
from data import DataLoader, natural_sort_key
from tensorboardX import SummaryWriter
#os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"

class Trainer():
    def __init__(self, args):
        self.args = args
        self.train_writer = SummaryWriter('Logs/train')
        self.test_writer = SummaryWriter('Logs/test')
        self.wavenet = Wavenet(args, self.train_writer)
        self.train_data_loader = DataLoader(
            args.batch_size * torch.cuda.device_count(), 
            args.shuffle, 
            args.num_workers, 
            True
        )
        self.test_data_loader = DataLoader(
            args.batch_size * torch.cuda.device_count(), 
            args.shuffle, 
            args.num_workers, 
            False
        )
        self.wavenet.total = self.train_data_loader.__len__() * self.args.num_epochs
        self.load_last_checkpoint(self.args.resume)
    
    def load_last_checkpoint(self, resume=0):
        if resume > 0:
            self.wavenet.load('Checkpoints/' + str(resume) + '_large.pkl', 'Checkpoints/' + str(resume) + '_small.pkl')
        else:
            checkpoint_list = list(pathlib.Path('Checkpoints').glob('**/*.pkl'))
            checkpoint_list = [str(i) for i in checkpoint_list]
            if len(checkpoint_list) > 0:
                checkpoint_list.sort(key=natural_sort_key)
                self.wavenet.load(str(checkpoint_list[-2]), str(checkpoint_list[-1]))

    def run(self):
        with tqdm(range(self.args.num_epochs), dynamic_ncols=True) as pbar1:
            for epoch in pbar1:
                with tqdm(self.train_data_loader, total=self.train_data_loader.__len__(), dynamic_ncols=True) as pbar2:
                    for i, (x, nonzero, diff, nonzero_diff, condition) in enumerate(pbar2):
                        step = i + epoch * self.train_data_loader.__len__()
                        current_large_loss, current_small_loss = self.wavenet.train(
                            x.cuda(non_blocking=True), 
                            nonzero.cuda(non_blocking=True), 
                            diff.cuda(non_blocking=True), 
                            nonzero_diff.cuda(non_blocking=True),
                            condition.cuda(non_blocking=True), 
                            step=step, train=True
                        )
                        pbar2.set_postfix(ll=current_large_loss, sl=current_small_loss)
                with torch.no_grad():
                    train_loss_large = train_loss_small = 0
                    with tqdm(self.test_data_loader, total=self.test_data_loader.__len__(), dynamic_ncols=True) as pbar2:
                        for x, nonzero, diff, nonzero_diff, condition in pbar2:
                            current_large_loss, current_small_loss = self.wavenet.train(
                                x.cuda(non_blocking=True), 
                                nonzero.cuda(non_blocking=True), 
                                diff.cuda(non_blocking=True), 
                                nonzero_diff.cuda(non_blocking=True),
                                condition.cuda(non_blocking=True), 
                                train=False
                            )
                            train_loss_large += current_large_loss
                            train_loss_small += current_small_loss
                            pbar2.set_postfix(ll=current_large_loss, sl=current_small_loss)
                    train_loss_large /= self.test_data_loader.__len__()
                    train_loss_small /= self.test_data_loader.__len__()
                    #tqdm.write('Testing step Large Loss: {}'.format(train_loss_large))
                    #tqdm.write('Testing step Small Loss: {}'.format(train_loss_small))
                    pbar1.set_postfix(ll=train_loss_large, sl=train_loss_small)
                    end_step = (epoch + 1) * self.train_data_loader.__len__()
                    sampled_image = self.sample(num=1, name=end_step)
                    self.test_writer.add_scalar('Test/Testing large loss', train_loss_large, end_step)
                    self.test_writer.add_scalar('Test/Testing small loss', train_loss_small, end_step)
                    self.test_writer.add_image('Score/Sampled', sampled_image, end_step)
                    self.wavenet.save(end_step)
        self.test_writer.close()
        self.train_writer.close()

    def sample(self, num, name='Sample_{}'.format(int(time.time()))):
        for _ in tqdm(range(num), dynamic_ncols=True):
            init, nonzero, diff, nonzero_diff, condition = self.train_data_loader.dataset.__getitem__(np.random.randint(self.train_data_loader.__len__()))
            image = self.wavenet.sample(
                name, 
                temperature=self.args.temperature, 
                init=torch.Tensor(init).cuda(non_blocking=True), 
                nonzero=torch.Tensor(nonzero).cuda(non_blocking=True), 
                diff=torch.Tensor(diff).cuda(non_blocking=True), 
                nonzero_diff=torch.Tensor(nonzero_diff).cuda(non_blocking=True), 
                condition=torch.Tensor(condition).cuda(non_blocking=True), 
                length=self.args.length
            )
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=7)
    parser.add_argument('--stack_size', type=int, default=2)
    parser.add_argument('--channels', type=int, default=326)
    parser.add_argument('--residual_channels', type=int, default=128)
    parser.add_argument('--dilation_channels', type=int, default=128)
    parser.add_argument('--skip_channels', type=int, default=128)
    parser.add_argument('--end_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=326)
    parser.add_argument('--condition_channels', type=int, default=6)
    parser.add_argument('--time_series_channels', type=int, default=7)
    parser.add_argument('--layer_size_small', type=int, default=9)
    parser.add_argument('--stack_size_small', type=int, default=2)
    parser.add_argument('--channels_small', type=int, default=326)
    parser.add_argument('--residual_channels_small', type=int, default=48)
    parser.add_argument('--dilation_channels_small', type=int, default=48)
    parser.add_argument('--skip_channels_small', type=int, default=48)
    parser.add_argument('--end_channels_small', type=int, default=48)
    parser.add_argument('--out_channels_small', type=int, default=7)
    parser.add_argument('--condition_channels_small', type=int, default=6)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--length', type=int, default=4096)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.)

    args = parser.parse_args()

    if args.sample > 0:
        args.batch_size = 1

    trainer = Trainer(args)

    if args.sample > 0:
        with torch.no_grad():
            trainer.sample(args.sample)
    else:
        trainer.run()
