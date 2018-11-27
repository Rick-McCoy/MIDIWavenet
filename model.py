import os
import torch
import torch.optim
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from data import INPUT_LENGTH, clean, piano_rolls_to_midi, save_roll
from network import Wavenet as WavenetModule

GEN_LENGTH = 4096

class Wavenet:
    def __init__(
            self, 
            layer_size, 
            stack_size, 
            channels, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            end_channels, 
            out_channels, 
            condition_channels, 
            lr, writer
        ):
        self.net = WavenetModule(
            layer_size, 
            stack_size, 
            channels, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            end_channels, 
            out_channels, 
            condition_channels
        )
        self.receptive_field = self.net.receptive_field
        self._prepare_for_gpu()
        self.channels = channels
        self.out_channels = out_channels
        self.lr = lr
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = self._loss()
        self.optimizer = self._optimizer()
        self.writer = writer
    
    def _loss(self):
        loss = torch.nn.BCELoss()
        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
    
    def _prepare_for_gpu(self):
        if torch.cuda.is_available():
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)

    def train(self, x, real, condition, step=1, train=True, total=0):
        output = self.net(x, condition).transpose(1, 2)[:, :-1, :]
        loss = self.loss(output.reshape(-1, self.out_channels), real.reshape(-1, self.out_channels))
        self.optimizer.zero_grad()
        if train:
            loss.backward()
            self.optimizer.step()
            if step % 20 == 19:
                tqdm.write('Training step {}/{} Loss: {}'.format(step, total, loss))
                self.writer.add_scalar('Training loss', loss.item(), step)
                self.writer.add_image('Real', real[:1], step)
                self.writer.add_image('Generated', output[:1], step)
        else:
            return loss.item()

    def sample(self, step, temperature=1., init=None, condition=None):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = self.generate(temperature, init, condition)
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
            for i in range(6):
                num = np.random.randint(0, 4)
                on = np.random.randint(0, 2)
                for j in range(num):
                    if on:
                        output[:, 324, -1] = 0
                        output[:, 324 + j] = 1
                        output[:, np.random.randint(channels[i], channels[i + 1]), -1] = 1
                        on = np.random.randint(0, 2)
        else:
            output[:, condition[:, 0] > 0] = 1
            for i, j in enumerate(condition):
                if j:
                    output[:, 324 + j] = 1
                    for _ in range(np.random.randint(0, 4)):
                        output[:, np.random.randint(channels[i], channels[i + 1]), -1] = 1
        return output

    def generate(self, temperature=1., init=None, condition=None):
        if init is None:
            init = self.gen_init(condition)
        else:
            init = np.expand_dims(init, axis=0)
            condition = np.expand_dims(condition, axis=0)
        init = init[:, :, -self.receptive_field - 2:-1] # pylint: disable=E1130
        output = np.zeros((self.out_channels, 1))
        self.net.module.fill_queues(torch.Tensor(init).cuda(), torch.Tensor(condition).cuda())
        x = init[:, :, -2:]
        for _ in tqdm(range(GEN_LENGTH)):
            nxt = self.net.module.sample_forward(torch.Tensor(x).cuda(), torch.Tensor(condition).cuda()).detach().cpu().numpy()
            if temperature != 1:
                nxt = np.power(nxt + 0.5, temperature) - 0.5
            nxt = nxt > np.random.rand(1, self.out_channels, 1)
            nxt = nxt.astype(np.float32)
            output = np.concatenate((output, nxt[0]), axis=1)
            x = np.concatenate((x, nxt), axis=2)
            x = x[:, :, 1:]
        return output[:, -GEN_LENGTH:]

    def save(self, step):
        if not os.path.exists('Checkpoints'):
            os.mkdir('Checkpoints')
        torch.save(self.net.state_dict(), 'Checkpoints/{}.pkl'.format(step))
    
    def load(self, path):
        tqdm.write('Loading from {}'.format(path))
        self.net.load_state_dict(torch.load(path))
