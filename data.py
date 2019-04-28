import os
import pathlib
import queue
from tqdm.autonotebook import tqdm
import pretty_midi as pm
import numpy as np
import torch
import warnings
import librosa.display
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
            event_list.append((round(note.start * 200), program, note.pitch + 129))
            event_list.append((round(note.end * 200), program, note.pitch + 257))
    event_list.sort()
    time_list = []
    current_time = 0
    for i in event_list:
        if current_time != i[0]:
            time_list.append(min(i[0] - current_time, 200) + 384)
            current_time = i[0]
        time_list.append(i[1])
        time_list.append(i[2])
    time_list += [585] * (INPUT_LENGTH - len(time_list))
    num = np.random.randint(0, len(time_list) - INPUT_LENGTH + 1)
    target = np.array(time_list[num : num + INPUT_LENGTH], dtype=np.int64) # pylint: disable=invalid-slice-index
    return condition, target

def clean(x):
    return x[0]

def save_roll(x, step):
    data = np.zeros((586, x.shape[0]))
    data[x, np.arange(x.shape[0])] = 1
    fig = plt.figure(figsize=(72, 24))
    librosa.display.specshow(data, x_axis='time', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
    plt.title('{}'.format(step))
    fig.savefig('Samples/{}.png'.format(step))
    plt.close(fig)

def piano_rolls_to_midi(x, fs=200):
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
                instruments[current_inst].notes.append(pm.Note(velocity=100, pitch=current_pitch, \
                                                                start=start, end=current_time))
        elif i < 585:
            time_incr = i - 384
            current_time += time_incr / fs
        else:
            break
    for inst in instruments:
        if inst.notes:
            midi.instruments.append(inst)
    return midi

class Dataset(data.Dataset):
    def __init__(self, train, length=None):
        super(Dataset, self).__init__()
        self.pathlist = np.array(train_list if train else test_list)
        self.len = length

    def __getitem__(self, index):
        while True:
            try:
                return midi_roll(np.random.choice(self.pathlist))
            except:
                continue

    def __len__(self):
        return self.pathlist.shape[0] if self.len is None else self.len

def init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=16, train=True, length=None):
        super(DataLoader, self).__init__(Dataset(train, length), \
                                            batch_size, shuffle, \
                                            num_workers=num_workers, \
                                            pin_memory=True, \
                                            worker_init_fn=init_fn)
