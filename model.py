import os
import torch
import itertools
import torch.optim
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from data import INPUT_LENGTH, clean, piano_rolls_to_midi, save_roll
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
            args.condition_channels
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
            args.condition_channels_small
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
        loss = torch.nn.BCELoss()
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
        mask = self.small_net(x, condition).transpose(1, 2)[:, :-1, :]
        output = self.large_net(x, condition).transpose(1, 2)[:, :-1, :]
        masked_output = output * diff
        masked_real = real * diff
        masked_output = masked_output.reshape(-1, self.out_channels)
        masked_real = masked_real.reshape(-1, self.out_channels)
        loss_large = self.loss(masked_output, masked_real)
        loss_small = self.loss(mask, diff)
        loss = loss_large + loss_small
        self.optimizer.zero_grad()
        if train:
            loss.backward()
            self.optimizer.step()
            if step % 20 == 19:
                tqdm.write('Training step {}/{} Loss: {}'.format(step, self.total, loss))
                self.writer.add_scalar('Training loss', loss.item(), step)
                self.writer.add_image('Real', real[:1], step)
                self.writer.add_image('Generated', output[:1], step)
        else:
            return loss.item()

    def sample(self, step, temperature=1., init=None, condition=None, length=2048):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = self.generate(temperature, init, condition, length)
        roll = clean(roll)
        save_roll(roll, step)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Saved to Samples/{}.mid'.format(step))
        roll = np.expand_dims(roll.T, axis=0)
        return roll

    def gen_init(self, condition=None):
        channels = [0, 72, 120, 192, 240, 288, 324]
        output = np.zeros([1, self.channels, self.receptive_field + 2])
        output[:, 324] = 1
        if condition is None:
            condition = np.random.randint(2, size=(6))
        output[:, condition[:, 0] > 0] = 1
        for i, j in enumerate(condition):
            if j:
                output[:, 324 + j] = 1
                for _ in range(np.random.randint(0, 4)):
                    output[:, np.random.randint(channels[i], channels[i + 1]), -1] = 1
        return output

    def generate(self, temperature=1., init=None, condition=None, length=2048):
        if init is None:
            init = self.gen_init(condition)
        else:
            init = np.expand_dims(init, axis=0)
            condition = np.expand_dims(condition, axis=0)
        init = init[:, :, :self.receptive_field + 2] # pylint: disable=E1130
        output = np.zeros((self.out_channels, 1))
        self.large_net.module.fill_queues(torch.Tensor(init).cuda(), torch.Tensor(condition).cuda())
        self.small_net.module.fill_queues(torch.Tensor(init).cuda(), torch.Tensor(condition).cuda())
        x = init[:, :, -2:]
        for _ in tqdm(range(length)):
            cont = self.small_net.module.sample_forward(torch.Tensor(x).cuda(), torch.Tensor(condition).cuda()).detach().cpu().numpy()
            if np.random.rand(1, 1, 1) > cont:
                output = np.concatenate((output, x[0, :, -1]), axis=1)
                x = np.concatenate((x, x[:, :, -1:]), axis=2)
                continue
            nxt = self.large_net.module.sample_forward(torch.Tensor(x).cuda(), torch.Tensor(condition).cuda()).detach().cpu().numpy()
            if temperature != 1:
                nxt = np.power(nxt + 0.5, temperature) - 0.5
            nxt = nxt > np.random.rand(1, self.out_channels, 1)
            nxt = nxt.astype(np.float32)
            output = np.concatenate((output, nxt[0]), axis=1)
            x = np.concatenate((x, nxt), axis=2)
            x = x[:, :, -2:]
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
