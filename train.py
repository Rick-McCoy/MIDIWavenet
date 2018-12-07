from __future__ import print_function
from __future__ import division
import os
import argparse
import torch
import pathlib
import time
import numpy as np
from tqdm import tqdm
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
            self.wavenet.receptive_field, 
            args.shuffle, 
            args.num_workers, 
            True
        )
        self.test_data_loader = DataLoader(
            args.batch_size * torch.cuda.device_count(), 
            self.wavenet.receptive_field, 
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
        for epoch in tqdm(range(self.args.num_epochs)):
            for i, (x, real, diff, condition) in enumerate(tqdm(self.train_data_loader, total=self.train_data_loader.__len__(), dynamic_ncols=True)):
                step = i + epoch * self.train_data_loader.__len__()
                self.wavenet.train(
                    x.cuda(), 
                    real.cuda(), 
                    diff.cuda(), 
                    condition.cuda(), 
                    step=step, train=True
                )
            with torch.no_grad():
                train_loss_large = train_loss_small = 0
                for _, (x, real, diff, condition) in enumerate(tqdm(self.test_data_loader, total=self.test_data_loader.__len__(), dynamic_ncols=True)):
                    current_large_loss, current_small_loss = self.wavenet.train(
                        x.cuda(), 
                        real.cuda(), 
                        diff.cuda(), 
                        condition.cuda(), 
                        train=False
                    )
                    train_loss_large += current_large_loss
                    train_loss_small += current_small_loss
                train_loss_large /= self.test_data_loader.__len__()
                train_loss_small /= self.test_data_loader.__len__()
                tqdm.write('Testing step Large Loss: {}'.format(train_loss_large))
                tqdm.write('Testing step Small Loss: {}'.format(train_loss_small))
                end_step = (epoch + 1) * self.train_data_loader.__len__()
                sampled_image = self.sample(num=1, name=end_step)
                self.test_writer.add_scalar('Test/Testing large loss', train_loss_large, end_step)
                self.test_writer.add_scalar('Test/Testing small loss', train_loss_small, end_step)
                self.test_writer.add_image('Score/Sampled', sampled_image, end_step)
                self.wavenet.save(end_step)

    def sample(self, num, name='Sample_{}'.format(int(time.time()))):
        for _ in tqdm(range(num), dynamic_ncols=True):
            init, _, diff, condition = self.train_data_loader.dataset.__getitem__(np.random.randint(self.train_data_loader.__len__()))
            image = self.wavenet.sample(
                name, 
                temperature=self.args.temperature, 
                init=torch.Tensor(init).cuda(), 
                diff=torch.Tensor(diff).cuda(), 
                condition=torch.Tensor(condition).cuda(), 
                length=self.args.length
            )
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=11)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--channels', type=int, default=326)
    parser.add_argument('--residual_channels', type=int, default=128)
    parser.add_argument('--dilation_channels', type=int, default=128)
    parser.add_argument('--skip_channels', type=int, default=512)
    parser.add_argument('--end_channels', type=int, default=512)
    parser.add_argument('--out_channels', type=int, default=326)
    parser.add_argument('--condition_channels', type=int, default=6)
    parser.add_argument('--time_series_channels', type=int, default=7)
    parser.add_argument('--channels_small', type=int, default=326)
    parser.add_argument('--residual_channels_small', type=int, default=64)
    parser.add_argument('--dilation_channels_small', type=int, default=64)
    parser.add_argument('--skip_channels_small', type=int, default=64)
    parser.add_argument('--end_channels_small', type=int, default=32)
    parser.add_argument('--out_channels_small', type=int, default=7)
    parser.add_argument('--condition_channels_small', type=int, default=6)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--length', type=int, default=2048)
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
