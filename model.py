import os
import torch
import itertools
import torch.optim
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from data import clean, piano_rolls_to_midi, save_roll
from network import Wavenet as WavenetModule

class Wavenet:
    def __init__(self, args, writer):
        self.large_net = WavenetModule(
            args.layer_size, 
            args.stack_size, 
            args.channels, 
            args.residual_channels, 
            args.dilation_channels, 
            args.skip_channels, 
            args.end_channels, 
            args.out_channels, 
            args.condition_channels, 
            args.time_series_channels
        )
        self.small_net = WavenetModule(
            args.layer_size_small, 
            args.stack_size_small, 
            args.channels_small, 
            args.residual_channels_small, 
            args.dilation_channels_small, 
            args.skip_channels_small, 
            args.end_channels_small, 
            args.out_channels_small, 
            args.condition_channels_small, 
            0
        )
        self.large_receptive_field = self.large_net.receptive_field
        self.small_receptive_field = self.small_net.receptive_field
        self._prepare_for_gpu()
        self.channels = args.channels
        self.out_channels = args.out_channels
        self.learning_rate = args.learning_rate
        self.large_loss = self._large_loss()
        self.small_loss = self._small_loss()
        self.optimizer_large = self._optimizer_large()
        self.optimizer_small = self._optimizer_small()
        self.writer = writer
        self.total = 0
    
    def _small_loss(self):
        loss = torch.nn.BCEWithLogitsLoss(
            # pos_weight=torch.cuda.FloatTensor([ # pylint: disable=E1101
            #     6.07501087e+01, 
            #     2.43404231e+02, 
            #     5.06094867e+02, 
            #     1.42857687e+03, 
            #     6.30166071e+02, 
            #     9.81309299e+02, 
            #     1.02464796e+00
            # ])
        )
        return loss

    def _large_loss(self):
        loss = torch.nn.BCEWithLogitsLoss()
        return loss

    def _optimizer_large(self):
        return torch.optim.Adam(self.large_net.parameters(), lr=self.learning_rate)

    def _optimizer_small(self):
        return torch.optim.Adam(self.small_net.parameters(), lr=self.learning_rate)

    def _prepare_for_gpu(self):
        if torch.cuda.is_available():
            self.large_net.cuda()
            self.large_net = torch.nn.DataParallel(self.large_net)
            self.small_net.cuda()
            self.small_net = torch.nn.DataParallel(self.small_net)

    def train(self, x, nonzero, diff, nonzero_diff, condition, length, step=1, train=True):
        mask = self.small_net(x[:, :, :-1], condition).transpose(1, 2)
        output = self.large_net(nonzero[:, :, :-1], condition, nonzero_diff.transpose(1, 2)).transpose(1, 2)
        diff = diff[:, -mask.shape[1]:]
        masked_output = [score[-ll:] for score, ll in zip(output, length)]
        masked_real = [score[-ll:] for score, ll in zip(nonzero.transpose(1, 2)[:, -output.shape[1]:], length)]
        loss_large = self.large_loss(torch.cat(masked_output, dim=0), torch.cat(masked_real, dim=0)) # pylint: disable=E1101
        loss_small = self.small_loss(mask.flatten(0, 1), diff.flatten(0, 1))
        loss_large_item = loss_large.item()
        loss_small_item = loss_small.item()
        if train:
            self.optimizer_large.zero_grad()
            loss_large.backward()
            self.optimizer_large.step()
            self.optimizer_small.zero_grad()
            loss_small.backward()
            self.optimizer_small.step()
            tqdm.write('Training step {}/{} Large loss: {}, Small loss: {}'.format(step, self.total, loss_large_item, loss_small_item))
            self.writer.add_scalar('Train/Large loss', loss_large_item, step)
            self.writer.add_scalar('Train/Small loss', loss_small_item, step)
            if step % 20 == 19:
                self.writer.add_image('Score/Real', masked_real[0].unsqueeze(dim=0), step)
                self.writer.add_image('Score/Generated', masked_output[0].unsqueeze(dim=0).sigmoid_(), step)
                self.writer.add_image('Hidden/Diff', diff[:1], step)
                self.writer.add_image('Hidden/Mask', mask[:1].sigmoid_(), step)
        del loss_large, loss_small
        return loss_large_item, loss_small_item

    def sample(self, step, temperature=1., init=None, nonzero=None, diff=None, nonzero_diff=None, condition=None, length=2048):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = self.generate(temperature, init, nonzero, nonzero_diff, condition, length).detach().cpu().numpy()
        roll = clean(roll)
        save_roll(roll, step)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Saved to Samples/{}.mid'.format(step))
        roll = np.expand_dims(roll.T, axis=0)
        return roll

    def gen_init(self, condition=None):
        channels = [0, 72, 120, 192, 240, 288, 324]
        output = torch.zeros([1, self.channels, self.large_receptive_field + 2]).cuda() # pylint: disable=E1101
        output[:, 324] = 1
        if condition is None:
            condition = torch.randint(2, size=(7,)).cuda() # pylint: disable=E1101
        for i, j in enumerate(condition):
            if j:
                for _ in range(np.random.randint(0, 4)):
                    output[:, np.random.randint(channels[i], channels[i + 1]), -1] = 1
        diff = torch.cat((output[:, :, 0:], output[:, :, -1:])) != output # pylint: disable=E1101
        return output, diff.to(torch.float32), condition # pylint: disable=E1101

    def generate(self, temperature=1., init=None, nonzero=None, nonzero_diff=None, condition=None, length=2048):
        if init is None:
            init, nonzero_diff, condition = self.gen_init(condition)
        else:
            init = init.unsqueeze(dim=0)
            nonzero = nonzero.unsqueeze(dim=0)
            nonzero_diff = nonzero_diff.unsqueeze(dim=0).transpose(1, 2)
            condition = condition.unsqueeze(dim=0)
        init = init[:, :, -self.small_receptive_field - 2:]
        nonzero = nonzero[:, :, -self.large_receptive_field - 2:]
        nonzero_diff = nonzero_diff[:, :, -self.large_receptive_field - 2:]
        self.large_net.module.fill_queues(nonzero, condition, nonzero_diff)
        self.small_net.module.fill_queues(init, condition)
        nonzero = nonzero[:, :, -2:]
        output = nonzero[0, :, -2:]
        nonzero_diff = nonzero_diff[:, :, -2:]
        for _ in tqdm(range(length), dynamic_ncols=True):
            cont = self.small_net.module.sample_forward(output[:, -2:].unsqueeze(dim=0), condition).sigmoid_()
            cont = torch.cuda.FloatTensor(cont.shape).uniform_() < cont # pylint: disable=E1101
            if cont.squeeze()[-1]:
                output = torch.cat((output, output[:, -1:]), dim=-1) # pylint: disable=E1101
                continue
            nonzero_diff = cont.to(torch.float32) # pylint: disable=E1101
            nxt = self.large_net.module.sample_forward(nonzero, condition, nonzero_diff) * temperature
            nxt = nxt.sigmoid_() > torch.cuda.FloatTensor(self.out_channels).uniform_() # pylint: disable=E1101
            nxt = nxt.to(torch.float32) # pylint: disable=E1101
            output = torch.cat((output, nxt[0]), dim=-1) # pylint: disable=E1101
            nonzero = torch.cat((nonzero, nxt), dim=-1) # pylint: disable=E1101
            nonzero = nonzero[:, :, -2:]
        return output[:, -length:]

    def save(self, step):
        if not os.path.exists('Checkpoints'):
            os.mkdir('Checkpoints')
        torch.save(self.large_net.state_dict(), 'Checkpoints/{}_large.pkl'.format(step))
        torch.save(self.small_net.state_dict(), 'Checkpoints/{}_small.pkl'.format(step))
    
    def load(self, path_large, path_small):
        tqdm.write('Loading from {}'.format(path_large))
        self.large_net.load_state_dict(torch.load(path_large))
        tqdm.write('Loading from {}'.format(path_small))
        self.small_net.load_state_dict(torch.load(path_small))
