import os
import pathlib
from tqdm.autonotebook import tqdm
import pretty_midi as pm
import numpy as np
import torch
import warnings
import librosa.display
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data

INPUT_LENGTH = 2048
NON_LENGTH = 512
with open('pathlist.txt', 'r') as f:
    pathlist = f.readlines()
pathlist = [x.strip() for x in pathlist]
#pathlist = list(pathlib.Path('Datasets/Classics').glob('**/*.mid')) + list(pathlib.Path('Datasets/Classics').glob('**/*.MID'))
#pathlist = list(pathlib.Path('Datasets/lmd_matched').glob('**/*.mid'))
np.random.shuffle(pathlist)
trainlist = pathlist[:-1024]
testlist = pathlist[-1024:]

def midi_roll(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path).replace('\\', '/'))
    event_list = []
    condition = np.zeros((128), dtype=np.bool)
    for inst in song.instruments:
        program = inst.program if not inst.is_drum else 9
        condition[program] = 1
        for note in inst.notes:
            event_list.append((int(note.start * song.resolution), program))
            event_list.append((int(note.start * song.resolution), note.pitch + 128))
            event_list.append((int(note.end * song.resolution), program))
            event_list.append((int(note.end * song.resolution), note.pitch + 256))
    event_list = list(set(event_list))
    event_list.sort()
    time_list = []
    wait_list = []
    wait_list_indice = []
    for i in event_list:
        if not time_list or time_list[-1][0] == i[0]:
            time_list.append(i)
        else:
            wait_list_indice.append(len(time_list))
            wait_list.append(i[0] - time_list[-1][0])
            time_list.append([i[0], i[0] - time_list[-1][0]])
    wait_list = np.array(wait_list, dtype=np.float32)
    norm = (wait_list - np.mean(wait_list)) / np.std(wait_list) if np.std(wait_list) else 0
    max_wait = np.amax(wait_list[norm < 5])
    wait_list = np.where(norm < 5, wait_list, max_wait)
    wait_list *= 100 / max_wait
    wait_list = wait_list.astype(np.int32)
    for i, j in zip(wait_list_indice, wait_list):
        time_list[i][1] = j + 384
    time_list = np.array([i[1] for i in time_list], dtype=np.int32)
    length = max(INPUT_LENGTH, time_list.shape[0])
    filler = length - time_list.shape[0]
    num = np.random.randint(0, length - INPUT_LENGTH + 1)
    time_list = time_list[num : num + INPUT_LENGTH]
    data = np.zeros((485, INPUT_LENGTH), dtype=np.bool)
    target = np.zeros((INPUT_LENGTH, ), dtype=np.int32)
    data[time_list, np.arange(time_list.shape[0])] = 1
    if filler:
        data[-1, -filler:] = 1
        target[-filler:] = 1
        target[:-filler] = time_list
    else:
        target = time_list
    return data.astype(np.float32), condition.astype(np.float32), target.astype(np.longlong)

def clean(x):
    return x.argmax(axis=1)

def save_roll(x, step):
    fig = plt.figure(figsize=(72, 24))
    librosa.display.specshow(x, x_axis='time', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
    plt.title('{}'.format(step))
    fig.savefig('Samples/{}.png'.format(step))
    plt.close(fig)

def piano_rolls_to_midi(x, fs=96):
    midi = pm.PrettyMIDI(resolution=fs)
    condition = [i in x for i in range(128)]
    instruments = [pm.Instrument(i, is_drum=i == 9) for i in range(128)]
    current_inst = current_pitch = current_time = 0
    start_time = np.zeros((128, 128))
    for i in x:
        if i < 128:
            current_inst = i
        elif i < 256:
            current_pitch = i - 128
            start_time[current_inst, current_pitch] = current_time
        elif i < 384:
            current_pitch = i - 256
            instruments[current_inst].notes.append(pm.Note(velocity=100, pitch=current_pitch, \
                                                            start=start_time[current_inst, current_pitch], \
                                                            end=current_time))
        elif i < 484:
            time_incr = i - 384
            current_time += time_incr / fs
    for i, j in enumerate(condition):
        if j:
            midi.instruments.append(instruments[i])
    return midi

class Dataset(data.Dataset):
    def __init__(self, train):
        super(Dataset, self).__init__()
        if train:
            self.pathlist = trainlist
        else:
            self.pathlist = testlist

    def __getitem__(self, index):
        return midi_roll(self.pathlist[index])

    def __len__(self):
        return len(self.pathlist)

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=16, train=True):
        super(DataLoader, self).__init__(Dataset(train), batch_size, shuffle, num_workers=num_workers, pin_memory=True)

def Test():
    len_list = []
    for i in tqdm(range(1000)):
        *_, length = midi_roll(pathlist[i])
        len_list.append(length)
    len_list.sort()
    plt.hist(len_list[:-100], bins=100, cumulative=True, histtype='step')
    plt.show()
    plt.close()

if __name__ == '__main__':
    Test()
