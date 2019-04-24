from __future__ import print_function
from __future__ import division
import os
import re
import argparse
import torch
import pathlib
import time
import warnings
from tqdm.autonotebook import tqdm
from model import Wavenet
from data import DataLoader
from tensorboardX import SummaryWriter

def natural_sort_key(s, _nsre=re.compile('(\\d+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

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
            True, 
            self.args.sample_step * args.batch_size * torch.cuda.device_count()
        )
        self.test_data_loader = DataLoader(
            args.batch_size * torch.cuda.device_count(), 
            args.shuffle, 
            args.num_workers, 
            False
        )
        self.train_range = self.train_data_loader.__len__()
        self.test_range = self.test_data_loader.__len__()
        self.start = 0
        self.start_1 = 0
        self.start_2 = 0
        self.load_last_checkpoint(self.args.resume)
    
    def load_last_checkpoint(self, resume=0):
        if resume > 0:
            self.wavenet.load('Checkpoints/' + str(resume) + '.pkl')
        else:
            checkpoint_list = list(pathlib.Path('Checkpoints').glob('**/*.pkl'))
            checkpoint_list = [str(i) for i in checkpoint_list]
            if len(checkpoint_list) > 0:
                checkpoint_list.sort(key=natural_sort_key)
                start = self.wavenet.load(str(checkpoint_list[-1]))
                self.start = start
                self.start_1 = start // self.train_data_loader.__len__()
                self.start_2 = start % self.train_data_loader.__len__()

    def run(self):
        step = self.start
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with tqdm(range(self.args.num_epochs), dynamic_ncols=True, initial=self.start_1) as pbar1:
                for epoch in pbar1:
                    if epoch and epoch % self.args.decay_accumulate == 0:
                        self.wavenet.accumulate *= 4
                    with tqdm(self.train_data_loader, total=self.train_range, dynamic_ncols=True, initial=self.start_2) as pbar2:
                        for condition, target in pbar2:
                            current_loss = self.wavenet.train(
                                condition.cuda(non_blocking=True), 
                                target.cuda(non_blocking=True), 
                                step=step, train=True
                            ).sum()
                            current_loss.backward()
                            if step % self.args.accumulate == self.args.accumulate - 1:
                                self.wavenet.optimizer.step()
                                self.wavenet.optimizer.zero_grad()
                            pbar2.set_postfix(loss=current_loss.item())
                            step += 1
                    with torch.no_grad():
                        test_loss = 0
                        with tqdm(self.test_data_loader, total=self.test_range, dynamic_ncols=True) as pbar3:
                            for condition, target in pbar3:
                                current_loss = self.wavenet.train(
                                    condition.cuda(non_blocking=True), 
                                    target.cuda(non_blocking=True), 
                                    train=False
                                ).sum().item()
                                test_loss += current_loss
                                pbar3.set_postfix(loss=current_loss)
                        test_loss /= self.test_range
                        pbar1.set_postfix(loss=test_loss)
                        sampled_image = self.sample(num=1, name=step)
                        self.test_writer.add_scalar('Test/Testing loss', test_loss, step)
                        self.test_writer.add_image('Score/Sampled', sampled_image, step)
                        self.wavenet.save(step)
        self.test_writer.close()
        self.train_writer.close()

    def sample(self, num, name='Sample_{}'.format(int(time.time()))):
        for _ in tqdm(range(num), dynamic_ncols=True):
            target, condition = self.train_data_loader.dataset.__getitem__(0)
            image = self.wavenet.sample(
                name, 
                temperature=self.args.temperature, 
                init=torch.Tensor(target).cuda(non_blocking=True), 
                condition=torch.Tensor(condition).cuda(non_blocking=True)
            )
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=11)
    parser.add_argument('--stack_size', type=int, default=2)
    parser.add_argument('--channels', type=int, default=586)
    parser.add_argument('--residual_channels', type=int, default=128)
    parser.add_argument('--dilation_channels', type=int, default=128)
    parser.add_argument('--skip_channels', type=int, default=128)
    parser.add_argument('--end_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=586)
    parser.add_argument('--condition_channels', type=int, default=129)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--decay_accumulate', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=1000)

    args = parser.parse_args()

    if args.sample > 0:
        args.batch_size = 1

    trainer = Trainer(args)

    if args.sample > 0:
        with torch.no_grad():
            trainer.sample(args.sample)
    else:
        trainer.run()
