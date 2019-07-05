import os
import pathlib
import queue
from tqdm.autonotebook import tqdm
import pretty_midi as pm
import numpy as np
import torch
import warnings
import time
import platform
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data

INPUT_LENGTH = 4096
with open('lmd_pathlist.txt', 'r') as f:
    pathlist = f.readlines()
pathlist = [x.strip() for x in pathlist]
with open('pathlist.txt', 'r') as f:
    add_pathlist = f.readlines()
pathlist += [x.strip() for x in add_pathlist]
np.random.shuffle(pathlist)
train_list = pathlist[:-1024]
test_list = pathlist[-1024:]

def midi_roll(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path).replace('\\', '/'))
    event_list = []
    condition = np.zeros((129), dtype=np.float32)
    for inst in song.instruments:
        program = inst.program if not inst.is_drum else 128
        condition[program] = 1
        for note in inst.notes:
            event_list.append((note.start * 1000, program * 256 + note.pitch))
            event_list.append((note.end * 1000, program * 256 + note.pitch + 128))
    event_list.sort()
    diff = np.diff(event_list, axis=0)[:, 0]
    diff = np.log10(diff[np.nonzero(diff)])
    diff_max = 4 - np.amax(diff)
    diff_min = -np.amin(diff)
    ratio = np.random.random() * max(diff_max + diff_min, 0) - diff_min
    time_list = [34024]
    current_time = event_list[0][0]
    for i in event_list:
        if i[0] > current_time:
            time_list.append(min(round((np.log10(i[0] - current_time) + ratio) * 250), 999) + 33024)
            current_time = i[0]
        time_list.append(i[1])
    time_list.append(34025)
    time_list += [34025] * (INPUT_LENGTH - len(time_list))
    num = np.random.randint(0, len(time_list) - INPUT_LENGTH + 1)
    target = np.array(time_list[num : num + INPUT_LENGTH], dtype=np.int64) # pylint: disable=invalid-slice-index
    return condition, target

def clean(x):
    x = x[0]
    time_list = []
    for i in x:
        if i < 33024:
            time_list.append(i // 256)
            time_list.append(i % 256 + 129)
        elif i < 34024:
            time_list.append(i - 33024 + 385)
        else:
            time_list.append(i - 34024 + 1385)
    return np.array(time_list)

def save_roll(x, step):
    data = np.zeros((1387, x.shape[0]))
    data[x, np.arange(x.shape[0])] = 1
    fig = plt.figure(figsize=(72, 24))
    plt.title('{}'.format(step))
    plt.imshow(data, origin='lower')
    fig.savefig('Samples/{}.png'.format(step), bbox_inches='tight', dpi=400)
    plt.close(fig)

def piano_rolls_to_midi(x, fs=1000):
    midi = pm.PrettyMIDI()
    instruments = [pm.Instrument(i) for i in range(128)] + [pm.Instrument(0, is_drum=True)]
    current_inst = current_pitch = current_time = 0
    start_time = [[queue.Queue()] * 128] * 129
    for i in x:
        if i < 129:
            current_inst = i
        elif i < 257:
            current_pitch = i - 129
            start_time[current_inst][current_pitch].put(current_time)
        elif i < 385:
            current_pitch = i - 257
            if not start_time[current_inst][current_pitch].empty():
                start = start_time[current_inst][current_pitch].get()
                if current_time != start:
                    instruments[current_inst].notes.append(pm.Note(velocity=100, pitch=current_pitch, \
                                                                    start=start, end=current_time))
        elif i < 1385:
            time_incr = i - 385
            current_time += np.power(10, time_incr / 250)
        elif i == 1386:
            break
    for inst in instruments:
        if inst.notes:
            midi.instruments.append(inst)
    return midi

class Dataset(data.Dataset):
    def __init__(self, train, length=None):
        super(Dataset, self).__init__()
        self.pathlist = np.array(train_list if train else test_list)
        self.length = length

    def __getitem__(self, index):
        while True:
            try:
                return midi_roll(np.random.choice(self.pathlist))
            except:
                continue

    def __len__(self):
        return self.pathlist.shape[0] if self.length is None else self.length

def init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=16, train=True, length=None):
        super(DataLoader, self).__init__(Dataset(train, length), \
                                            batch_size, shuffle, \
                                            num_workers=num_workers, \
                                            pin_memory=True, \
                                            worker_init_fn=init_fn)
