"""Main file for training. Contains Trainer, a training class for Wavenet.
All hyperparameters are modifiable via flags."""
from __future__ import print_function
from __future__ import division

import argparse
import time
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from model import Wavenet
from data import DataLoader
from utils import get_checkpoint

class Trainer():
    """The training class. Initialized by args from flags.
    Can load from checkpoint & run for a set number of epochs.

    Arguments
    -----------
    args : Namespace
        A collection of arguments used to initialize Wavenet.

    Parameters
    -----------
    args : Namespace
        A collection of arguments used to initialize Wavenet.

    train_writer : torch.utils.tensorboard.SummaryWriter
        SummaryWriter for tensorboard, used to record values from training.

    test_writer : torch.utils.tensorboard.SummaryWriter
        SummaryWriter for tensorboard, used to record values from testing.

    wavenet : WavenetModule
        WavenetModule from model.py.

    train_data_loader : DataLoader
        DataLoader from data.py, used for training.

    test_data_loader : DataLoader
        DataLoader from data.py, used for testing.

    train_range : int
        Length of train_data_loader.

    start_1 : int
        Epoch to resume from.

    start_2 : int
        Step in first epoch to resume from.

    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.train_writer = SummaryWriter('Logs/train')
        self.test_writer = SummaryWriter('Logs/test')
        self.wavenet = Wavenet(args, self.train_writer)
        self.train_data_loader = DataLoader(
            batch_size=args.batch_size * torch.cuda.device_count(),
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            train=True,
            input_length=self.args.output_length + self.wavenet.receptive_field + 1,
            output_length=self.args.output_length,
            dataset_length=self.args.sample_step * args.batch_size * torch.cuda.device_count()
        )
        self.test_data_loader = DataLoader(
            batch_size=args.batch_size * torch.cuda.device_count(),
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            train=False,
            input_length=self.args.output_length + self.wavenet.receptive_field + 1,
            output_length=self.args.output_length
        )
        self.start_1 = 0
        self.start_2 = 0
        self.load_last_checkpoint(self.args.resume)

    def load_last_checkpoint(self, resume=0):
        """Loads last checkpoint. Calls get_checkpoint(resume), then attempts to load from it.
        If get_checkpoint failed, goes on without loading.

        Arguments
        ------------
        resume : int
            Name of the checkpoint to resume from, 0 if starting from scratch.

        Returns
        ------------
        Does not return anything."""
        checkpoint = get_checkpoint(resume)
        if checkpoint is not None:
            self.wavenet.load(checkpoint)
            self.start_1 = self.wavenet.count // len(self.train_data_loader)
            self.start_2 = self.wavenet.step % len(self.train_data_loader)
            self.train_data_loader.dataset.dataset_length *= self.wavenet.accumulate

    def run(self):
        """Runs training schemes for given number of epochs & sample steps.
        Will generate samples periodically, after each epoch is finished.
        tqdm progress bars features current loss without having to print every step.
        Features nested progress bars; Expect buggy behavior.

        Arguments
        -----------
        No arguments passed.

        Returns
        -----------
        Does not return anything."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with tqdm(
                    range(self.args.num_epochs),
                    dynamic_ncols=True,
                    initial=self.start_1
            ) as pbar1:
                for epoch in pbar1:
                    if self.args.increase_batch_size and epoch \
                        and epoch % self.args.increase_batch_size == 0:
                        self.wavenet.accumulate *= 2
                        self.train_data_loader.dataset.dataset_length *= 2
                        tqdm.write('Accumulate = {}'.format(self.wavenet.accumulate))
                    with tqdm(
                            self.train_data_loader,
                            dynamic_ncols=True,
                            initial=self.start_2
                    ) as pbar2:
                        for target, condition in pbar2:
                            current_loss = self.wavenet.train(
                                target=target.cuda(non_blocking=True),
                                condition=condition.cuda(non_blocking=True),
                                output_length=self.args.output_length
                            )
                            pbar2.set_postfix(loss=current_loss)
                    self.start_2 = 0
                    with torch.no_grad():
                        test_loss = []
                        with tqdm(self.test_data_loader, dynamic_ncols=True) as pbar3:
                            for target, condition in pbar3:
                                current_loss = self.wavenet.get_loss(
                                    target=target.cuda(non_blocking=True),
                                    condition=condition.cuda(non_blocking=True),
                                    output_length=self.args.output_length
                                ).item()
                                test_loss.append(current_loss)
                                pbar3.set_postfix(loss=current_loss)
                        test_loss = sum(test_loss) / len(test_loss)
                        pbar1.set_postfix(loss=test_loss)
                        sampled_image = self.sample(num=1, name=self.wavenet.step)
                        self.write_test_loss(loss=test_loss, image=sampled_image)
                        self.wavenet.save()
        self.test_writer.close()
        self.train_writer.close()

    def write_test_loss(self, loss, image):
        self.test_writer.add_scalar(
            tag='Test/Test loss count',
            scalar_value=loss,
            global_step=self.wavenet.count
        )
        self.test_writer.add_scalar(
            tag='Test/Test loss',
            scalar_value=loss,
            global_step=self.wavenet.step
        )
        self.test_writer.add_image(
            tag='Score/Sampled',
            img_tensor=image,
            global_step=self.wavenet.step
        )

    def sample(self, num, name='Sample_{}'.format(int(time.time()))):
        """Samples from trained Wavenet. Can specify number of samples & name of each.

        Arguments
        ------------
        num: int
            Number of samples to be generated.

        name: string
            Name of sample to be generated.

        Returns
        ------------
        image: np.array
            2d piano roll representation of last generated sample."""
        for _ in tqdm(range(num), dynamic_ncols=True):
            target, condition = self.train_data_loader.dataset.__getitem__(0)
            image = self.wavenet.sample(
                name=name,
                init=torch.from_numpy(target).cuda(non_blocking=True),
                condition=torch.from_numpy(condition).cuda(non_blocking=True),
                temperature=self.args.temperature
            )
        return image

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--layer_size', type=int, default=10)
    PARSER.add_argument('--stack_size', type=int, default=4)
    PARSER.add_argument('--channels', type=int, default=128)
    PARSER.add_argument('--embedding_channels', type=int, default=512)
    PARSER.add_argument('--residual_channels', type=int, default=256)
    PARSER.add_argument('--dilation_channels', type=int, default=256)
    PARSER.add_argument('--skip_channels', type=int, default=512)
    PARSER.add_argument('--end_channels', type=int, default=1024)
    PARSER.add_argument('--condition_channels', type=int, default=256)
    PARSER.add_argument('--kernel_size', type=int, default=5)
    PARSER.add_argument('--learning_rate', type=float, default=1e-3)
    PARSER.add_argument('--resume', type=int, default=0)
    PARSER.add_argument('--sample', type=int, default=0)
    PARSER.add_argument('--temperature', type=float, default=1.)
    PARSER.add_argument('--batch_size', type=int, default=16)
    PARSER.add_argument('--accumulate', type=int, default=1)
    PARSER.add_argument('--increase_batch_size', type=int, default=4)
    PARSER.add_argument('--num_epochs', type=int, default=100)
    PARSER.add_argument('--sample_step', type=int, default=1024)
    PARSER.add_argument('--shuffle', type=bool, default=True)
    PARSER.add_argument('--num_workers', type=int, default=1)
    PARSER.add_argument('--output_length', type=int, default=1024)

    PARSED_ARGS = PARSER.parse_args()

    if PARSED_ARGS.sample > 0:
        PARSED_ARGS.batch_size = 1

    TRAINER = Trainer(PARSED_ARGS)

    if PARSED_ARGS.sample > 0:
        with torch.no_grad():
            TRAINER.sample(PARSED_ARGS.sample)
    else:
        TRAINER.run()
