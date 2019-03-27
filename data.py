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
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data

INPUT_LENGTH = 8192
with open('lmd_pathlist.txt', 'r') as f:
    pathlist = f.readlines()
pathlist = [x.strip() for x in pathlist]
with open('pathlist.txt', 'r') as f:
    add_pathlist = f.readlines()
pathlist += [x.strip() for x in add_pathlist]
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
    condition = np.zeros((129), dtype=np.bool)
    for inst in song.instruments:
        program = inst.program if not inst.is_drum else 128
        condition[program] = 1
        for note in inst.notes:
            event_list.append((int(note.start * 200), program, note.pitch + 129))
            event_list.append((int(note.end * 200), program, note.pitch + 257))
    event_list.sort()
    time_list = []
    for i in event_list:
        if time_list and time_list[-1][0] != i[0]:
            time_list.append([time_list[-1][0], min(i[0] - time_list[-1][0], 200) + 385])
        time_list.append(i[:2])
        time_list.append([i[0], i[2]])
    time_list = np.array([i[-1] for i in time_list], dtype=np.int32)
    length = max(INPUT_LENGTH, time_list.shape[0])
    filler = length - time_list.shape[0]
    num = np.random.randint(0, length - INPUT_LENGTH + 1)
    time_list = time_list[num : num + INPUT_LENGTH]
    data = np.zeros((586, INPUT_LENGTH), dtype=np.bool)
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
    return x[0].argmax(axis=0)

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
            time_incr = i - 385
            current_time += time_incr / fs
    for i in range(129):
        if instruments[i].notes:
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
        while True:
            try:
                path = self.pathlist[np.random.randint(0, len(self.pathlist))]
                return midi_roll(path)
            except:
                tqdm.write(path)
                continue

    def __len__(self):
        return len(self.pathlist)

def init_fn(worker_id):
    np.random.RandomState(worker_id + torch.initial_seed()) # pylint: disable=E1101

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=16, train=True):
        super(DataLoader, self).__init__(Dataset(train), \
                                            batch_size, shuffle, \
                                            num_workers=num_workers, \
                                            pin_memory=True, \
                                            worker_init_fn=init_fn)

def Test():
    # print(len(pathlist))
    # len_list = []
    # for _ in tqdm(range(200)):
    #     while True:
    #         try:
    #             length = midi_roll(pathlist[np.random.randint(0, len(pathlist))])
    #             break
    #         except:
    #             continue
    #     len_list.append(length)
    # len_list.sort()
    # plt.hist(len_list[:-20], bins=100, cumulative=True, histtype='step')
    # plt.show()
    # plt.close()
    song = midi_roll('Datasets/lmd_matched/A/X/L/TRAXLZU12903D05F94/1bddc5dbd78f2d242a02a9985bc6b400.mid')
    midi = piano_rolls_to_midi(song)
    midi.write('Samples/Never.mid')

if __name__ == '__main__':
    Test()
