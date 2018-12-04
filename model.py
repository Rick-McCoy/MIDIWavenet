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
        self.receptive_field = self.large_net.receptive_field
        self._prepare_for_gpu()
        self.channels = args.channels
        self.out_channels = args.out_channels
        self.learning_rate = args.learning_rate
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = self._loss()
        self.optimizer = self._optimizer()
        self.writer = writer
        self.total = 0
    
    def _loss(self):
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([48.5]).cuda(non_blocking=True))
        return loss

    def _optimizer(self):
        return torch.optim.Adam(itertools.chain(self.large_net.parameters(), self.small_net.parameters()), lr=self.learning_rate)
    
    def _prepare_for_gpu(self):
        if torch.cuda.is_available():
            self.large_net.cuda()
            self.large_net = torch.nn.DataParallel(self.large_net)
            self.small_net.cuda()
            self.small_net = torch.nn.DataParallel(self.small_net)

    def train(self, x, real, diff, condition, step=1, train=True):
        mask = self.small_net(x, condition).transpose(1, 2)
        output = self.large_net(x, condition, diff.transpose(1, 2)).transpose(1, 2)
        diff = diff[:, -output.shape[1]:]
        writer_mask = mask[:1].sigmoid_()
        writer_diff = diff[:1]
        mask = mask.reshape(-1, 6)
        diff = diff.reshape(-1, 6)
        indices = diff.sum(dim=1).nonzero()
        masked_output = output.reshape(-1, self.out_channels)[indices]
        masked_real = real.reshape(-1, self.out_channels)[indices]
        loss_large = self.loss(masked_output, masked_real)
        loss_small = self.loss(mask, diff)
        loss = loss_large + loss_small
        self.optimizer.zero_grad()
        if train:
            loss.backward()
            self.optimizer.step()
            if step % 20 == 19:
                tqdm.write('Training step {}/{} Large loss: {}, Small loss: {}'.format(step, self.total, loss_large.item(), loss_small.item()))
                self.writer.add_scalar('Train/Large loss', loss_large.item(), step)
                self.writer.add_scalar('Train/Small loss', loss_small.item(), step)
                self.writer.add_image('Score/Real', real[:1], step)
                self.writer.add_image('Score/Generated', output[:1].sigmoid_(), step)
                self.writer.add_image('Hidden/Mask', writer_mask, step)
                self.writer.add_image('Hidden/Diff', writer_diff, step)
        return loss_large.item(), loss_small.item()

    def sample(self, step, temperature=1., init=None, diff=None, condition=None, length=2048):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = self.generate(temperature, init, diff, condition, length).detach().cpu().numpy()
        roll = clean(roll)
        save_roll(roll, step)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Saved to Samples/{}.mid'.format(step))
        roll = np.expand_dims(roll.T, axis=0)
        return roll

    def gen_init(self, condition=None):
        channels = [0, 72, 120, 192, 240, 288, 324]
        output = torch.zeros([1, self.channels, self.receptive_field + 2]).cuda(non_blocking=True) # pylint: disable=E1101
        output[:, 324] = 1
        if condition is None:
            condition = torch.randint(2, size=(6,)).cuda(non_blocking=True) # pylint: disable=E1101
        for i, j in enumerate(condition):
            if j:
                for _ in range(np.random.randint(0, 4)):
                    output[:, np.random.randint(channels[i], channels[i + 1]), -1] = 1
        diff = torch.cat((output[:, :, 0:], output[:, :, -1:])) != output # pylint: disable=E1101
        return output, diff.to(torch.float32), condition # pylint: disable=E1101

    def generate(self, temperature=1., init=None, diff=None, condition=None, length=2048):
        if init is None:
            init, diff, condition = self.gen_init(condition)
        else:
            init.unsqueeze_(dim=0)
            diff = diff.unsqueeze_(dim=0).transpose(1, 2)
            condition.unsqueeze_(dim=0)
        init = init[:, :, :self.receptive_field + 2]
        output = torch.zeros((self.out_channels, 1)).cuda(non_blocking=True) # pylint: disable=E1101
        self.large_net.module.fill_queues(init, condition, diff)
        self.small_net.module.fill_queues(init, condition)
        x = init[:, :, -2:]
        for _ in tqdm(range(length)):
            cont = torch.cuda.FloatTensor(1, 6, 1).uniform_() < self.small_net.module.sample_forward(x[:, :, -2:], condition) # pylint: disable=E1101
            cont = cont.to(torch.float32) # pylint: disable=E1101
            diff = torch.cat((diff, cont), dim=-1) # pylint: disable=E1101
            if cont.sum() == 0:
                output = torch.cat((output, x[0, :, -1:]), dim=-1) # pylint: disable=E1101
                x = torch.cat((x, x[:, :, -1:]), dim=-1) # pylint: disable=E1101
                continue
            diff = diff[:, :, 1 - x.shape[2]:]
            nxt = self.large_net.module.sample_forward(x, condition, diff)
            if temperature != 1:
                nxt += 0.5
                nxt.pow(temperature)
                nxt -= 0.5
            nxt = nxt > torch.cuda.FloatTensor(self.out_channels).uniform_() # pylint: disable=E1101
            nxt = nxt.to(torch.float32) # pylint: disable=E1101
            output = torch.cat((output, nxt[0]), dim=-1) # pylint: disable=E1101
            x = torch.cat((x, nxt), dim=-1) # pylint: disable=E1101
            x = x[:, :, -2:]
            del cont, nxt
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
