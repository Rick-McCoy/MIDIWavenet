"""Wavenet, the raw network itself.
Contains various components of it as well as the network.
No hardcoded values, all passed arguments can be changed without issues."""
import queue
from torch import nn, cat, zeros
from torch.utils.checkpoint import checkpoint
from utils import causal_pad

class DilatedConv1d(nn.Module):
    """Dilated 1D Convolution, the basis for Wavenet.
    Supports sampling mode which disables dilation for fast generation.
    As per paper, the convolution layer does not have a bias parameter.

    Arguments
    ----------
    in_channels : int
        Number of channels of input.

    out_channels : int
        Number of channels of output.

    kernel_size : int
        Size of kernel.

    dilaiton : int
        Length of dilation.

    Parameters
    -----------
    conv : torch.nn.Conv1d
        Torch layer for convolution while training.

    sample_conv : torch.nn.Conv1d
        Torch layer for convolution while sampling.
        Shares weight with sample_conv.

    Methods
    -----------
    forward

        Arguments
        -----------
        x : torch.Tensor
            Input sequence.

        sample : bool
            Determines sampling.

        Returns
        -----------
        self.conv(x) : torch.Tensor
            Performs convolution on input x.
            If sample=True, returns self.sample_conv(x).
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=False
        )
        self.sample_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=False
        )
        self.sample_conv.weight = self.conv.weight

    def forward(self, x, sample=False):
        return self.sample_conv(x) if sample else self.conv(x)

class ResidualBlock(nn.Module):
    """The basic residual block.
    Consists of convolution layers for the input sequence & condition.
    For efficiency, condition convolution layers are implemented as linear layers.

    Arguments
    -----------
    residual_channels : int
        Number of channels of input sequence & residual sum.

    dilation_channels : int
        Number of channels of intermediate layers.

    skip_channels : int
        Number of channels of accumulated output.

    condition_channels : int
        Number of channels of condition.

    kernel_size : int
        Size of kernel of dilated convolutions.

    dilation : int
        Length of dilation.

    Parameters
    ------------
    dilation : int
        Length of dilation.

    filter_conv : DilatedConv1d
        Convolution layer for input sequence, activation function tanh.

    gate_conv : DilatedConv1d
        Convolution layer for input sequence, activation function sigmoid.

    filter_linear : torch.nn.Linear
        Convolution layer for condition, implemented as torch.nn.Linear.
        Activation funciton tanh.

    gate_linear : torch.nn.Linear
        Convolution layer for condition, implemented as torch.nn.Linear.
        Activation funciton sigmoid.

    residual_conv : torch.nn.Conv1d
        Convolution layer for residual sum.

    skip_conv : torch.nn.Conv1d
        Convolution layer for accumulated output.

    Methods
    ----------
    forward

        Arguments
        ------------
        x : torch.Tensor
            Input sequence.

        condition : torch.Tensor
            Global condition.

        res_sum : torch.Tensor
            Residual sum tensor.

        sample : bool
            Boolean to determine sampling.

        Returns
        ------------
        output : torch.Tensor
            Output tensor of ResidualBlock.

        res_sum : torch.Tensor
            Accumulated residual sum.
    """
    def __init__(
            self,
            residual_channels,
            dilation_channels,
            skip_channels,
            condition_channels,
            kernel_size,
            dilation
    ):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.filter_conv = DilatedConv1d(
            residual_channels,
            dilation_channels,
            kernel_size,
            dilation
        )
        self.gate_conv = DilatedConv1d(
            residual_channels,
            dilation_channels,
            kernel_size,
            dilation
        )
        self.filter_linear = nn.Linear(
            condition_channels,
            dilation_channels
        )
        self.gate_linear = nn.Linear(
            condition_channels,
            dilation_channels
        )
        self.residual_conv = nn.Conv1d(
            dilation_channels,
            residual_channels,
            1
        )
        self.skip_conv = nn.Conv1d(
            dilation_channels,
            skip_channels,
            1
        )
        self.queues = [
            queue.Queue(dilation + 1) for _ in range(kernel_size - 1)
        ]
        self.output_length = 1
        self.conditional_filter = 0
        self.conditional_gate = 0

    def forward(self, x, condition=0, res_sum=0, sample=False):
        dilated_filter = self.filter_conv(x, sample)
        dilated_gate = self.gate_conv(x, sample)
        if sample:
            conditional_filter = self.conditional_filter
            conditional_gate = self.conditional_gate
        else:
            conditional_filter = self.filter_linear(condition).unsqueeze(dim=-1)
            conditional_gate = self.gate_linear(condition).unsqueeze(dim=-1)
        dilated_filter += conditional_filter
        dilated_gate += conditional_gate
        dilated_filter.tanh_()
        dilated_gate.sigmoid_()
        dilated = dilated_filter * dilated_gate
        output = self.residual_conv(dilated) + x[..., -dilated.shape[2]:]
        res_sum = res_sum + self.skip_conv(dilated)[..., -self.output_length:]
        return output, res_sum

    def set_condition(self, condition):
        """Sets condition for sampling mode.
        Condition is reused during entire duration of sampling,
        thus meaningless to recalculate filter & gate every time."""
        self.conditional_filter = self.filter_linear(condition).unsqueeze(dim=-1)
        self.conditional_gate = self.gate_linear(condition).unsqueeze(dim=-1)

class ResidualStack(nn.Module):
    """Stack of ResidualBlocks: Has no layers of its own.
    Handles residual summation and sampling.

    Arguments
    -----------
    layer_size : int
        Size of layer, determines exponential part of receptive field.

    stack_size : int
        Size of stack, determines linear part of receptive field.
        Repeats layer stack_size times.

    residual_channels : int
        Number of channels of input sequence & residual sum.

    dilation_channels : int
        Number of channels of intermediate layers within ResidualBlock.

    skip_channels : int
        Number of channels of accumulated output.

    condition_channels : int
        Number of channels of global condition.

    kernel_size : int
        Size of kernel of dilated convolutions.

    Parameters
    -----------
    dilations : list
        List of dilations for each ResidualBlock.

    res_blocks : torch.nn.ModuleList
        ModuleList of ResidualBlocks.

    Methods
    -----------
    forward

        Arguments
        -----------

        target : torch.Tensor
            Input sequence.

        condition : torch.Tensor
            Global condition.

        output_length : int
            Length of output tensor.

        Returns
        ----------
        res_sum : torch.Tensor
            Residual sum of all ResidualBlocks.
    """
    def __init__(
            self,
            layer_size,
            stack_size,
            residual_channels,
            dilation_channels,
            skip_channels,
            condition_channels,
            kernel_size
        ):
        super(ResidualStack, self).__init__()
        self.dilations = [2 ** i for i in range(layer_size)] * stack_size
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels,
                dilation_channels,
                skip_channels,
                condition_channels,
                kernel_size,
                dilation
            ) for dilation in self.dilations
        ])

    def forward(self, target, condition, output_length):
        res_sum = zeros((1, 1, 1), device=target.device)
        for res_block in self.res_blocks:
            res_block.output_length = output_length
            target, res_sum = checkpoint(res_block, target, condition, res_sum)
        return res_sum

    def sample_forward(self, target):
        """Sampling function, operates at O(stack_size * layer_size * kernel_size).
        Samples one time step at a time.
        All queues of each individual ResidualBlock needs to be filled first before sampling.

        Arguments
        -----------
        target : torch.Tensor
            Input sequence, to be extended by sampling.

        Returns
        -----------
        res_sum : torch.Tensor
            Residual sum of all ResidualBlocks."""
        res_sum = 0
        for res_block in self.res_blocks:
            res_block.output_length = 1
            tops = [target] + [que.get() for que in res_block.queues]
            for que, top in zip(res_block.queues, tops[:-1]):
                que.put(top)
            target = cat(tops[::-1], dim=-1)
            target, res_sum = res_block(target, res_sum=res_sum, sample=True)
        return res_sum

    def fill_queues(self, target, condition):
        """Prepares ResidualBlock for sampling mode.
        Calls set_condition and fills queues for each res_block.

        Arguments
        -----------
        target : torch.Tensor
            Input sequence for filling queues.

        condition : torch.Tensor
            Global condition for set_condition.

        Returns
        ----------
        Does not return anything."""
        for res_block in self.res_blocks:
            res_block.output_length = 1
            for i, que in enumerate(res_block.queues):
                with que.mutex:
                    que.queue.clear()
                for j in range(-res_block.dilation, 0):
                    que.put(target[..., -res_block.dilation * i + j - 1].unsqueeze(dim=-1))
            res_block.set_condition(condition)
            target, _ = res_block(target, sample=True)

class PostProcess(nn.Module):
    """Simple Post processing, contains two convolutions.
    Inplace relu is used for efficiency.

    Arguments
    -----------
    skip_channels : int
        Number of channels of accumulated outputs.

    end_channels : int
        Number of channels of intermediate layer.

    channels : int
        Number of channels of output.

    Parameters
    -----------
    conv1 : torch.nn.Conv1d
        First convolution.

    conv2 : torch.nn.Conv1d
        Second convolution.

    relu : torch.nn.ReLU
        Inplace ReLU activation function.

    Methods
    -----------
    forward

        Arguments
        -----------
        target : torch.Tensor
            Input residual sum.

        Returns
        -----------
        output : torch.Tensor
            Output tensor."""
    def __init__(self, skip_channels, end_channels, channels):
        super(PostProcess, self).__init__()
        self.conv1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.conv2 = nn.Conv1d(end_channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, target):
        output = self.relu(target)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        return output

class Wavenet(nn.Module):
    """Base module of Wavenet.
    Supports sampling via Fast Wavenet.
    Forward function calculates loss, for output call get_output.

    Arguments
    -----------
    layer_size : int
        Size of layer, determines exponential part of receptive field.

    stack_size : int
        Size of stack, determines linear part of receptive field.

    channels : int
        Number of channels of input sequence & output.

    embedding_channels : int
        Number of channels of output of embedding.

    residual_channels : int
        Number of channels of residual sum.

    skip_channels : int
        Number of channels of accumulated outputs.

    end_channels : int
        Number of channels of intermediate layer in PostProcess.

    condition_channels : int
        Number of channels of global condition.

    kernel_size : int
        Size of kernel of dilated convolutions.

    Parameters
    ------------
    receptive_field : int
        Receptive field of res_stsack.

    embedding : torch.nn.Embedding
        Embedding layer for token to embedding vector conversion.

    causal : DilatedConv1d
        Causal convolution layer for temporal alignment.

    res_stacks : ResidualStack
        ResidualStack for dilated convolutions.

    post : PostProcess
        PostProcess for final convolution layers.

    loss : torch.nn.CrossEntropyLoss
        CrossEntropyLoss for loss calculation.

    Methods
    ----------
    forward

        Arguments
        -----------
        target : torch.Tensor
            Input sequence.

        condition : torch.Tensor
            Global condition.

        output_length : int
            Length of output.

        Returns
        -----------
        loss : torch.Tensor
            CrossEntropyLoss of output.
    """
    def __init__(
            self,
            layer_size,
            stack_size,
            channels,
            embedding_channels,
            residual_channels,
            dilation_channels,
            skip_channels,
            end_channels,
            condition_channels,
            kernel_size
        ):
        super(Wavenet, self).__init__()
        self.receptive_field = (2 ** layer_size - 1) * stack_size * (kernel_size - 1)
        self.embedding = nn.Embedding(channels, embedding_channels)
        self.causal = DilatedConv1d(
            embedding_channels,
            residual_channels,
            kernel_size=2,
            dilation=1
        )
        self.res_stacks = ResidualStack(
            layer_size,
            stack_size,
            residual_channels,
            dilation_channels,
            skip_channels,
            condition_channels,
            kernel_size
        )
        self.post = PostProcess(skip_channels, end_channels, channels)
        self.loss = nn.CrossEntropyLoss()

    def get_output(self, target, condition, output_length):
        """Returns raw output from Wavenet.
        Specify output length.

        Arguments
        -----------
        target : torch.Tensor
            Input sequence.

        condition : torch.Tensor
            Global condition.

        output_length : int
            Length of output.

        Returns
        ----------
        output : torch.Tensor
            Output of Wavenet."""
        target = target[..., :-1]
        output = self.embedding(target).transpose(1, 2)
        output = causal_pad(output)
        output = self.causal(output)
        output = self.res_stacks(output, condition, output_length)
        output = self.post(output)
        return output

    def forward(self, target, condition, output_length):
        output = self.get_output(target, condition, output_length)
        loss = self.loss(output, target[:, -output_length:])
        return loss

    def sample_output(self, target):
        """Output function used for sampling purposes.
        Global condition must be primed with fill_queues.

        Arguments
        -----------
        target : torch.Tensor
            Input snippet.

        Returns
        -----------
        output : torch.Tenosr
            Output time step."""
        output = self.embedding(target).transpose(1, 2)
        output = causal_pad(output)
        output = self.causal(output)[..., 1:]
        output = self.res_stacks.sample_forward(output)
        output = self.post(output)
        return output

    def fill_queues(self, target, condition):
        """Global condition & queue primer function.
        Fills queues of ResidualStack with input sequence.
        Global condition is reused throughout task, improving speed.

        Arguments
        -----------
        target : torch.Tensor
            Input sequence.

        condition : torch.Tensor
            Global condition.

        Returns
        -----------
        Does not return anything."""
        target = self.embedding(target).transpose(1, 2)
        target = causal_pad(target)
        target = self.causal(target)
        self.res_stacks.fill_queues(target, condition)
