import torch
import queue
import numpy as np
from torch.utils.checkpoint import checkpoint

class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, padding=1, 
                                    dilation=1, bias=False)

    def forward(self, x, dummy=None):
        return self.conv(x)[:, :, :-1]

class DilatedCausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, dilation=dilation, 
                                    bias=False)
        self.sample_conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, bias=False)
        self.sample_conv.weight = self.conv.weight

    def forward(self, x, sample=False):
        return self.sample_conv(x) if sample else self.conv(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, condition_channels, dilation, time_series_channels):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.filter_conv = DilatedCausalConv1d(residual_channels, dilation_channels, dilation=dilation)
        self.gate_conv = DilatedCausalConv1d(residual_channels, dilation_channels, dilation=dilation)
        self.conditional_filter_conv = torch.nn.Conv1d(condition_channels, dilation_channels, 1)
        self.conditional_gate_conv = torch.nn.Conv1d(condition_channels, dilation_channels, 1)
        self.residual_conv = torch.nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = torch.nn.Conv1d(dilation_channels, skip_channels, 1)
        self.queue = queue.Queue(dilation)
        if time_series_channels > 0:
            self.time_filter_conv = torch.nn.Conv1d(time_series_channels, dilation_channels, 1)
            self.time_gate_conv = torch.nn.Conv1d(time_series_channels, dilation_channels, 1)
        self.time_on = time_series_channels > 0
        self.skip_size = 1

    def forward(self, x, condition, time_series=None, sample=False):
        dilated_filter = self.filter_conv(x, sample)
        dilated_gate = self.gate_conv(x, sample)
        conditional_filter = self.conditional_filter_conv(condition)
        conditional_gate = self.conditional_gate_conv(condition)
        dilated_filter += conditional_filter
        if self.time_on:
            time_series_filter = self.time_filter_conv(time_series)
            dilated_filter += time_series_filter
        dilated_filter.tanh_()
        dilated_gate += conditional_gate
        if self.time_on:
            time_series_gate = self.time_gate_conv(time_series)
            dilated_gate += time_series_gate
        dilated_gate.sigmoid_()
        dilated = dilated_filter * conditional_gate

        output = self.residual_conv(dilated)
        output += x[:, :, -output.shape[2]:]

        skip = self.skip_conv(dilated)[:, :, -self.skip_size:]
        return output, skip

class ResidualStack(torch.nn.Module):
    def __init__(
            self, 
            layer_size, 
            stack_size, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            condition_channels, 
            time_series_channels
        ):
        super(ResidualStack, self).__init__()
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.dilations = [2 ** i for i in range(self.layer_size)] * self.stack_size
        self.res_blocks = torch.nn.ModuleList(
            self.stack_res_blocks(
                residual_channels, 
                dilation_channels, 
                skip_channels, 
                condition_channels, 
                time_series_channels
            )
        )
        self.time_on = time_series_channels > 0

    def stack_res_blocks(self, residual_channels, dilation_channels, skip_channels, condition_channels, time_series_channels):
        res_blocks = [ResidualBlock(
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            condition_channels, 
            dilation, 
            time_series_channels
        ) for dilation in self.dilations]
        return res_blocks

    def forward(self, x, condition, skip_size, time_series=None):
        output = x
        res_sum = 0
        for res_block, dilation in zip(self.res_blocks, self.dilations):
            res_block.skip_size = skip_size
            if self.time_on:
                time_series = time_series[:, :, dilation:]
                output, skip = checkpoint(res_block, output, condition, time_series)
            else:
                output, skip = checkpoint(res_block, output, condition)
            res_sum += skip
            #del skip
        return res_sum

    def sample_forward(self, x, condition, time_series=None):
        output = x
        res_sum = 0
        for res_block in self.res_blocks:
            res_block.skip_size = 1
            for i in range(output.shape[2]):
                top = res_block.queue.get()
                current = output[:, :, i].unsqueeze(dim=-1)
                res_block.queue.put(current)
                full = torch.cat((top, current), dim=-1) # pylint: disable=E1101
                if self.time_on:
                    current_time = time_series[:, :, i].unsqueeze(dim=-1)
                    current, skip = res_block(full, condition, current_time, sample=True)
                else:
                    current, skip = res_block(full, condition, sample=True)
                output[:, :, i] = current.squeeze(dim=-1)
                #del current_time, top, full, current
            res_sum += skip
            #del skip
        return res_sum

    def fill_queues(self, x, condition, time_series=None):
        for res_block, dilation in zip(self.res_blocks, self.dilations):
            res_block.skip_size = 1
            if self.time_on:
                time_series = time_series[:, :, dilation:]
            with res_block.queue.mutex:
                res_block.queue.queue.clear()
            for i in range(-res_block.dilation - 1, -1):
                res_block.queue.put(x[:, :, i:i + 1])
            x, _ = res_block(x, condition, time_series)

class PostProcess(torch.nn.Module):
    def __init__(self, skip_channels, end_channels, out_channels):
        super(PostProcess, self).__init__()
        self.conv1 = torch.nn.Conv1d(skip_channels, end_channels, 1)
        self.conv2 = torch.nn.Conv1d(end_channels, out_channels, 1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.relu(x)
        #output = self.conv1(output)
        output = checkpoint(self.conv1, output)
        output = self.relu(output)
        output = self.conv2(output)
        return output
    
    def sample_forward(self, x):
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
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            end_channels, 
            out_channels, 
            condition_channels, 
            time_series_channels
        ):
        super(Wavenet, self).__init__()
        self.receptive_field = self.calc_receptive_field(layer_size, stack_size)
        self.causal = CausalConv1d(channels, residual_channels)
        self.res_stacks = ResidualStack(
            layer_size, 
            stack_size, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            condition_channels, 
            time_series_channels
        )
        self.post = PostProcess(skip_channels, end_channels, out_channels)

    @staticmethod
    def calc_receptive_field(layer_size, stack_size):
        layers = [2 ** i for i in range(layer_size)] * stack_size
        return sum(layers)

    def calc_output_size(self, x):
        output_size = x.size()[2] - self.receptive_field
        return output_size

    def forward(self, x, condition, time_series=None):
        output_size = self.calc_output_size(x)
        #output = self.causal(x)
        dummy = torch.zeros_like(condition, requires_grad=True) # pylint: disable=E1101
        output = checkpoint(self.causal, x, dummy)
        output = self.res_stacks(output, condition, output_size, time_series)
        output = self.post(output)
        #del dummy
        return output[:, :, :-1]

    def sample_forward(self, x, condition, time_series=None):
        #output = self.causal(x)[:, :, 1:]
        output = self.causal(x)[:, :, 1:]
        output = self.res_stacks.sample_forward(output, condition, time_series)
        output = self.post.sample_forward(output)
        return output.sigmoid_()

    def fill_queues(self, x, condition, time_series=None):
        x = self.causal(x)
        self.res_stacks.fill_queues(x, condition, time_series)
