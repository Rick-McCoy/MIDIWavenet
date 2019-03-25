import os
import torch
import itertools
import torch.optim
import numpy as np
import pretty_midi as pm
from tqdm.autonotebook import tqdm
from data import clean, piano_rolls_to_midi, save_roll
from network import Wavenet as WavenetModule

class Wavenet:
    def __init__(self, args, writer):
        self.net = WavenetModule(
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
        self.receptive_field = self.net.receptive_field
        self._prepare_for_gpu()
        self.channels = args.channels
        self.out_channels = args.out_channels
        self.learning_rate = args.learning_rate
        self.loss = self._loss()
        self.optimizer = self._optimizer()
        self.writer = writer
        self.total = 0

    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()
        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def _prepare_for_gpu(self):
        if torch.cuda.is_available():
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)

    def train(self, x, condition, target, step=1, train=True):
        output = self.net(x[:, :, :-1], condition)
        loss = self.loss(output, target[:, -output.shape[2]:])
        loss_item = loss.item()
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('Train/Loss', loss_item, step)
            if step % 20 == 19:
                self.writer.add_image('Score/Real', x[0, :, -output.shape[2]:].unsqueeze(dim=0), step)
                self.writer.add_image('Score/Generated', torch.nn.functional.softmax(output[0].unsqueeze(dim=0), dim=1), step)
        del loss
        return loss_item

    def sample(self, step, temperature=1., init=None, condition=None):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        image_roll = self.generate(temperature, init, condition).detach().cpu().numpy()
        roll = clean(image_roll)
        save_roll(roll, step)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Saved to Samples/{}.mid'.format(step))
        return image_roll

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
        return output, condition # pylint: disable=E1101

    def generate(self, temperature=1., init=None, condition=None):
        if init is None:
            init = self.gen_init(condition)
        else:
            init = init.unsqueeze(dim=0)
        init = init[:, :, -self.receptive_field - 2:]
        self.net.module.fill_queues(init, condition)
        output = init[..., -2:]
        for i in tqdm(range(10000), dynamic_ncols=True):
            cont = self.net.module.sample_forward(output[..., -2:], condition)
            cont = torch.nn.functional.softmax(cont, dim=1)
            output = torch.cat((output, cont), dim=-1) # pylint: disable=E1101
            if i % 20 == 0 and cont.squeeze().detach().cpu().numpy().argmax() == cont.squeeze().shape[0] - 1:
                break
        return output[..., 1:]

    def save(self, step):
        if not os.path.exists('Checkpoints'):
            os.mkdir('Checkpoints')
        torch.save(self.net.state_dict(), 'Checkpoints/{}.pkl'.format(step))
    
    def load(self, path):
        tqdm.write('Loading from {}'.format(path))
        self.net.load_state_dict(torch.load(path))
