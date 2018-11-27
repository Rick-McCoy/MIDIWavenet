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
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"

class Trainer():
    def __init__(self, args):
        self.args = args
        self.train_writer = SummaryWriter('Logs/train')
        self.test_writer = SummaryWriter('Logs/test')
        self.wavenet = Wavenet(
            args.layer_size, 
            args.stack_size, 
            args.channels, 
            args.residual_channels, 
            args.dilation_channels, 
            args.skip_channels, 
            args.end_channels, 
            args.out_channels, 
            args.condition_channels, 
            args.learning_rate, 
            self.train_writer
        )
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
    
    def load_last_checkpoint(self):
        checkpoint_list = list(pathlib.Path('Checkpoints').glob('**/*.pkl'))
        checkpoint_list = [str(i) for i in checkpoint_list]
        if len(checkpoint_list) > 0:
            checkpoint_list.sort(key=natural_sort_key)
            self.wavenet.load(str(checkpoint_list[-1]))

    def run(self):
        self.load_last_checkpoint()
        for epoch in tqdm(range(self.args.num_epochs)):
            for i, (sample, real, condition) in tqdm(enumerate(self.train_data_loader), total=self.train_data_loader.__len__()):
                step = i + epoch * self.train_data_loader.__len__()
                self.wavenet.train(sample.cuda(), real.cuda(), condition.cuda(), step, True, self.args.num_epochs * self.train_data_loader.__len__())
            with torch.no_grad():
                train_loss = 0
                for _, (sample, real, condition) in tqdm(enumerate(self.test_data_loader), total=self.test_data_loader.__len__()):
                    train_loss += self.wavenet.train(sample.cuda(), real.cuda(), condition.cuda(), train=False)
                train_loss /= self.test_data_loader.__len__()
                tqdm.write('Testing step Loss: {}'.format(train_loss))
                end_step = (epoch + 1) * self.train_data_loader.__len__()
                sample_init, _, sample_condition = self.train_data_loader.dataset.__getitem__(np.random.randint(self.train_data_loader.__len__()))
                sampled_image = self.wavenet.sample(end_step, init=sample_init, condition=sample_condition)
                self.test_writer.add_scalar('Testing loss', train_loss, end_step)
                self.test_writer.add_image('Sampled', sampled_image, end_step)
                self.wavenet.save(end_step)

    def sample(self, num):
        self.load_last_checkpoint()
        with torch.no_grad():
            for _ in tqdm(range(num)):
                sample_init, _, sample_condition = self.train_data_loader.dataset.__getitem__(np.random.randint(self.train_data_loader.__len__()))
                self.wavenet.sample('Sample_{}'.format(int(time.time())), temperature=self.args.temperature, init=sample_init, condition=sample_condition)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=10)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--channels', type=int, default=326)
    parser.add_argument('--residual_channels', type=int, default=256)
    parser.add_argument('--dilation_channels', type=int, default=512)
    parser.add_argument('--skip_channels', type=int, default=512)
    parser.add_argument('--end_channels', type=int, default=1024)
    parser.add_argument('--out_channels', type=int, default=326)
    parser.add_argument('--condition_channels', type=int, default=6)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.)

    args = parser.parse_args()

    if args.sample > 0:
        args.batch_size = 1

    trainer = Trainer(args)

    if args.sample > 0:
        trainer.sample(args.sample)
    else:
        trainer.run()
