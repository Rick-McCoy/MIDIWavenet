import torch
import queue
import numpy as np
from torch.utils.checkpoint import checkpoint

class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, bias=False)

    def forward(self, x):
        output = torch.nn.functional.pad(x, (1, 0), 'constant')
        output = self.conv(output)
        return output

class DilatedCausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=kernel_size, dilation=dilation, 
                                    bias=False)
        self.sample_conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=kernel_size, bias=False)
        self.sample_conv.weight = self.conv.weight

    def forward(self, x, sample=False):
        return self.sample_conv(x) if sample else self.conv(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, condition_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.filter_conv = DilatedCausalConv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation)
        self.gate_conv = DilatedCausalConv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation)
        self.conditional_filter_linear = torch.nn.Linear(condition_channels, dilation_channels)
        self.conditional_gate_linear = torch.nn.Linear(condition_channels, dilation_channels)
        self.residual_conv = torch.nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = torch.nn.Conv1d(dilation_channels, skip_channels, 1)
        self.queues = [queue.Queue(dilation + 1) for _ in range(kernel_size - 1)]
        self.skip_size = 1

    def forward(self, x, condition, res_sum, sample=False):
        dilated_filter = self.filter_conv(x, sample)
        dilated_gate = self.gate_conv(x, sample)
        conditional_filter = self.conditional_filter_linear(condition).unsqueeze(dim=-1)
        conditional_gate = self.conditional_gate_linear(condition).unsqueeze(dim=-1)
        dilated_filter += conditional_filter
        dilated_gate += conditional_gate
        dilated_filter.tanh_()
        dilated_gate.sigmoid_()
        dilated = dilated_filter * dilated_gate
        output = self.residual_conv(dilated)
        output += x[..., -output.shape[2]:]
        res_sum = res_sum + self.skip_conv(dilated)[..., -self.skip_size:]
        return output, res_sum

class ResidualStack(torch.nn.Module):
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
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.skip_channels = skip_channels
        self.dilations = [2 ** i for i in range(self.layer_size)] * self.stack_size
        self.res_blocks = torch.nn.ModuleList([
            ResidualBlock(
                residual_channels, 
                dilation_channels, 
                skip_channels, 
                condition_channels, 
                kernel_size, 
                dilation
            ) for dilation in self.dilations
        ])

    def forward(self, x, condition, skip_size):
        res_sum = torch.zeros((1, 1, 1), device=x.device) # pylint: disable=no-member
        for res_block in self.res_blocks:
            res_block.skip_size = skip_size
            x, res_sum = checkpoint(res_block, x, condition, res_sum)
        return res_sum

    def sample_forward(self, x, condition):
        res_sum = 0
        for res_block in self.res_blocks:
            res_block.skip_size = 1
            top = x[..., -1:]
            for que in res_block.queues:
                que.put(top)
                top = que.get()
                x = torch.cat((top, x), dim=-1) # pylint: disable=no-member
            x, res_sum = res_block(x, condition, res_sum, sample=True)
        return res_sum

    def fill_queues(self, x, condition):
        for res_block in self.res_blocks:
            res_block.skip_size = 1
            for i, que in enumerate(res_block.queues):
                with que.mutex:
                    que.queue.clear()
                for j in range(res_block.dilation):
                    que.put(x[..., -res_block.dilation * (i + 1) + j - 1].unsqueeze(dim=-1))
            x, _ = res_block(x, condition, 0)

class PostProcess(torch.nn.Module):
    def __init__(self, skip_channels, end_channels, channels):
        super(PostProcess, self).__init__()
        self.conv1 = torch.nn.Conv1d(skip_channels, end_channels, 1)
        self.conv2 = torch.nn.Conv1d(end_channels, channels, 1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        return output

class Wavenet(torch.nn.Module):
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
        self.receptive_field =  (2 ** layer_size - 1) * stack_size * (kernel_size - 1)
        self.embedding = torch.nn.Embedding(channels, embedding_channels)
        self.causal = CausalConv1d(embedding_channels, residual_channels)
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
        self.loss = torch.nn.CrossEntropyLoss()
    
    def get_output(self, target, condition):
        x = target[..., :-1]
        output_size = x.shape[-1] - self.receptive_field
        output = self.embedding(x).transpose(1, 2)
        output = self.causal(output)
        output = self.res_stacks(output, condition, output_size)
        output = self.post(output)
        return output

    def forward(self, target, condition):
        output = self.get_output(target, condition)
        loss = self.loss(output, target[:, -output.shape[-1]:])
        return loss

    def sample_output(self, x, condition):
        output = self.embedding(x).transpose(1, 2)
        output = self.causal(output)[..., 1:]
        output = self.res_stacks.sample_forward(output, condition)
        output = self.post(output)
        return output

    def fill_queues(self, x, condition):
        x = self.embedding(x).transpose(1, 2)
        x = self.causal(x)
        self.res_stacks.fill_queues(x, condition)
