import os
import torch
import itertools
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
            args.embedding_channels, 
            args.residual_channels, 
            args.dilation_channels, 
            args.skip_channels, 
            args.end_channels, 
            args.condition_channels, 
            args.kernel_size
        )
        self.receptive_field = self.net.receptive_field
        self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)
        self.channels = args.channels
        self.learning_rate = args.learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()
        self.writer = writer
        self.accumulate = args.accumulate
        self.loss_sum = 0

    def train(self, condition, target, step=1, train=True):
        loss = self.net(target, condition).sum() / self.accumulate / torch.cuda.device_count()
        if train:
            loss.backward()
            self.loss_sum += loss.item()
            if step % self.accumulate == self.accumulate - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.writer.add_scalar('Train/Loss', self.loss_sum, step)
                self.loss_sum = 0
            if step % 500 == 499:
                output = self.net.module.get_output(target[:1], condition[:1])
                x = clean(target[..., -output.shape[2]:].detach().cpu().numpy())
                image = np.zeros((1, 1387, x.shape[0]))
                image[0, x, np.arange(x.shape[0])] = 1
                self.writer.add_image('Score/Real', image, step)
                x = clean(output.argmax(dim=1).detach().cpu().numpy())
                image = np.zeros((1, 1387, x.shape[0]))
                image[0, x, np.arange(x.shape[0])] = 1
                self.writer.add_image('Score/Generated', image, step)
        return loss.item() * self.accumulate

    def sample(self, step, init=None, condition=None, temperature=1):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        image_roll = self.generate(init, condition, temperature).detach().cpu().numpy()
        roll = clean(image_roll)
        save_roll(roll, step)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Saved to Samples/{}.mid'.format(step))
        sampled_image = np.zeros((1, 1387, roll.shape[0]))
        sampled_image[0, roll, np.arange(roll.shape[0])] = 1
        return sampled_image

    def generate(self, init, condition, temperature=1.):
        init = init.unsqueeze(dim=0)[..., -self.receptive_field - 2:]
        self.net.module.fill_queues(init, condition)
        output = init
        for i in tqdm(range(10000), dynamic_ncols=True):
            cont = self.net.module.sample_output(output[..., -2:], condition).squeeze().softmax(dim=0)
            cont = self.top_p(cont)
            token = torch.multinomial(cont, num_samples=1).unsqueeze(dim=0) # pylint: disable=no-member
            output = torch.cat((output, token), dim=-1) # pylint: disable=no-member
            if i % 20 == 0 and token.item() == self.channels - 1:
                break
        return output
    
    def top_p(self, logits, perc=0.9):
        values, indice = torch.sort(logits, dim=-1, descending=True) # pylint: disable=no-member
        cumsum = torch.cumsum(values, dim=-1) # pylint: disable=no-member
        point = torch.sum(cumsum > perc).item() # pylint: disable=no-member
        logits[..., indice[..., 1 - point:]] = 0
        return logits

    def save(self, step):
        if not os.path.exists('Checkpoints'):
            os.mkdir('Checkpoints')
        state = {
            'model': self.net.state_dict(), 
            'step': step + 1, 
            'optimizer': self.optimizer.state_dict(), 
            'accumulate': self.accumulate
        }
        torch.save(state, 'Checkpoints/{}.pkl'.format(step))
    
    def load(self, path):
        tqdm.write('Loading from {}'.format(path))
        load = torch.load(path)
        self.net.load_state_dict(load['model'])
        self.optimizer.load_state_dict(load['optimizer'])
        self.accumulate = load['accumulate']
        return load['step'] - 1
