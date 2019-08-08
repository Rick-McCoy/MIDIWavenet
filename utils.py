import pathlib
import re
import platform
import numpy as np
import torch
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def top_p(logits: torch.Tensor, perc=0.9) -> torch.Tensor:
    values, indice = torch.sort(logits, dim=-1, descending=True)
    cumsum = torch.cumsum(values, dim=-1)
    point = torch.sum(cumsum > perc).item()
    logits[..., indice[..., 1 - point:]] = 0
    return logits

def to_image(target: np.ndarray) -> np.ndarray:
    image = np.zeros((1, 128, target.shape[0]))
    image[0, target, np.arange(target.shape[0])] = 1
    return image

def clean(target: torch.Tensor) -> np.ndarray:
    return target.detach().cpu().numpy()[0]

def get_score(output: torch.Tensor, answer: torch.Tensor) -> np.ndarray:
    output = clean(output)
    score = np.broadcast_to(output[answer, np.arange(output.shape[-1])], output.shape)
    return np.expand_dims(score, axis=0)

def save_roll(target: np.ndarray, step: int):
    data = np.zeros((128, target.shape[0]))
    data[target, np.arange(target.shape[0])] = 1
    fig = plt.figure(figsize=(72, 24))
    plt.title('{}'.format(step))
    plt.imshow(data, origin='lower')
    fig.savefig('Samples/{}.png'.format(step), bbox_inches='tight', dpi=400)
    plt.close(fig)

def get_checkpoint(resume: int) -> str:
    if resume:
        return 'Checkpoints/' + str(resume) + '.pkl'
    checkpoint_list = list(pathlib.Path('Checkpoints').glob('**/*.pkl'))
    checkpoint_list = [str(i) for i in checkpoint_list]
    if checkpoint_list:
        checkpoint_list.sort(key=natural_sort_key)
        return checkpoint_list[-1]
    return None

def natural_sort_key(target: str, _nsre=re.compile('(\\d+)')) -> list:
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(target)]

def causal_pad(target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(target, (1, 0), 'constant')

def init_fn(worker_id: int):
    np.random.seed(torch.initial_seed() % (2 ** 32) + worker_id)
