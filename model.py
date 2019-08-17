"""Contains class Wavenet, the middle ground between the raw module & the trainer.
Useful interfaces such as train, sample, save, load are implemented."""
import os
import torch
from tqdm.autonotebook import tqdm
from data import piano_rolls_to_midi
from utils import top_p, to_image, clean, save_roll, get_accuracy
from network import Wavenet as WavenetModule

class Wavenet:
    """The middle ground between the module & the trainer.
    Provides interfaces for the trainer to use.

    Arguments
    -----------
    args : Namespace
        A collection of parameters used to initialize Wavenet.

    writer : torch.utils.tensorboard.SummaryWriter
        SummaryWriter for tensorboard.target

    Parameters
    -----------
    net : WavenetModule
        The Wavenet itself. Wrapped in Dataparallel, so access with net.module.

    receptive_field : int
        Receptive field of Wavenet.

    optimizer : torch.optim.adam.Adam
        Optimizer for Wavenet.

    writer : torch.utils.tensorboard.SummaryWriter
        SummaryWriter for tensorboard.

    accumulate : int
        Accumulate for gradient accumulation.

    loss_sum : float
        Internal variable for keeping track of loss while accumulating.

    step : int
        Number of backpropagations performed.

    count : int
        Number of parameter updates performed.
    """
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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.optimizer.zero_grad()
        self.writer = writer
        self.accumulate = args.accumulate
        self.loss_sum = 0
        self.step = 0
        self.count = 0

    def get_loss(self, target, condition, output_length=1):
        loss = self.net(target, condition, output_length).sum()
        return loss / torch.cuda.device_count()

    def write_loss(self):
        self.writer.add_scalar('Train/Train loss count', self.loss_sum, self.count)
        self.loss_sum = 0

    def take_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.count += 1

    def write_image(self, target, condition, output_length):
        output = self.net.module.get_output(target[:1], condition[:1], output_length)
        cleaned_input = clean(target[..., -output_length:])
        self.writer.add_image('Score/Input', to_image(cleaned_input), self.step)
        cleaned_output = clean(output.argmax(dim=1))
        self.writer.add_image('Score/Output', to_image(cleaned_output), self.step)
        score, accuracy = get_accuracy(torch.nn.functional.softmax(output, dim=2), cleaned_input)
        self.writer.add_histogram('Score/Accuracy', score, self.step)
        self.writer.add_image('Score/Score', score, self.step)
        self.writer.add_scalar('Train/Accuracy', accuracy, self.step)

    def train(self, target: torch.Tensor, condition: torch.Tensor, output_length=1) -> torch.Tensor:
        """Training method; Calculates gradients for given input and returns loss.
        Takes optimizer step every self.accumulate steps.
        Keeps track of count.

        Arguments
        --------------
        condition : Tensor
            Global condition for Wavenet.

        target : Tensor
            Main input for Wavenet.

        output_length : int
            Length of output of Wavenet.

        Returns
        --------------
        loss : float
            Loss of Wavenet's output. Adjusted for accumulation.
        """
        loss = self.get_loss(target, condition, output_length) / self.accumulate
        loss.backward()
        self.loss_sum += loss.item()
        self.writer.add_scalar('Train/Training Loss', loss.item() * self.accumulate, self.step)
        self.step += 1
        if self.step % self.accumulate == 0:
            self.write_loss()
            self.take_step()
        if self.step % 100 == 0:
            self.write_image(target[:1], condition[:1], output_length)
        return loss.item() * self.accumulate

    def sample(self, name: str, init: torch.Tensor, condition: torch.Tensor, temperature=1.):
        """Sampling: Wrapper around generate, handles saving to midi & saving roll as plot.

        Arguments
        --------------
        name : int or str
            The name of the resulting sampled midi file.

        init : Tensor or None
            Initializing tensor for Wavenet in order for fast generation.
            Currently None is not supported.

        condition : Tensor or None
            Condition tensor for Wavenet.
            Currently None is not supported.

        temperature : float
            Sampling temperature; >1 means more randomness, <1 means less randomness.

        Returns
        --------------
        to_image(roll) : np.array
            2d piano roll representation of generated sample.
        """
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = clean(self.generate(init, condition, temperature))
        save_roll(roll, name)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(name))
        tqdm.write('Saved to Samples/{}.mid'.format(name))
        return to_image(roll)

    def generate(self, init: torch.Tensor, condition: torch.Tensor, temperature=1.) -> torch.Tensor:
        """Generating: Samples from Wavenet using top_p sampling.

        Arguments
        ------------
        init : Tensor
            Initializing tensor for Wavenet.

        condition : Tensor
            Condition tensor for Wavenet.

        temperature : float
            Sampling temperature.

        Returns
        ------------
        output : Tensor
            Sampled output from Wavenet.
        """
        init = init.unsqueeze(dim=0)[..., -self.receptive_field - 1:]
        self.net.module.fill_queues(init, condition)
        output = init
        for _ in tqdm(range(10000), dynamic_ncols=True):
            cont = self.net.module.sample_output(output[..., -2:]) * temperature
            cont = cont.squeeze().softmax(dim=0)
            cont = top_p(cont)
            token = torch.multinomial(cont, num_samples=1)
            token = token.unsqueeze(dim=0)
            output = torch.cat((output, token), dim=-1)
        return output

    def save(self):
        """Saving method: Generating checkpoints while training.

        Arguments
        -----------
        No arguments passed.

        Returns
        -----------
        Does not return anything.
        """
        if not os.path.exists('Checkpoints'):
            os.mkdir('Checkpoints')
        state = {
            'model': self.net.state_dict(),
            'step': self.step,
            'count': self.count,
            'optimizer': self.optimizer.state_dict(),
            'accumulate': self.accumulate
        }
        torch.save(state, 'Checkpoints/{}.pkl'.format(self.step))

    def load(self, path: str):
        """Loading method: Loads from checkpoints.

        Arguments
        -----------
        path : string
            Path to checkpoint to load from.

        Returns
        -----------
        Does not return anything.
        """
        tqdm.write('Loading from {}'.format(path))
        load = torch.load(path)
        self.net.load_state_dict(load['model'])
        self.optimizer.load_state_dict(load['optimizer'])
        self.accumulate = load['accumulate']
        self.step = load['step']
        self.count = load['count']
