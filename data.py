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
import random
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
    current_time = 0
    for i in event_list:
        if current_time != i[0]:
            time_list.append(min(i[0] - current_time, 200) + 384)
            current_time = i[0]
        time_list.append(i[1])
        time_list.append(i[2])
    time_list = np.array(time_list, dtype=np.longlong)
    length = max(INPUT_LENGTH, time_list.shape[0])
    filler = length - time_list.shape[0]
    num = np.random.randint(0, length - INPUT_LENGTH + 1)
    time_list = time_list[num : num + INPUT_LENGTH]
    target = np.zeros((INPUT_LENGTH, ), dtype=np.longlong)
    if filler > 0:
        target[-filler:] = 585
        target[:-filler] = time_list
    else:
        target = time_list
    return condition.astype(np.float32), target

def test_function(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path).replace('\\', '/'))
    # note_list = []
    # for inst in song.instruments:
    #     note_list += inst.notes
    # dur_list = [(note.end - note.start) * 1000 for note in note_list]
    # sort_list = [(note.start * 1000, note.end * 1000) for note in note_list]
    # sort_list.sort()
    # time_list = list(np.diff(np.array([i[0] for i in sort_list])))
    # return dur_list, time_list
    return len(song.instruments), sum([inst.is_drum for inst in song.instruments])

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
            time_incr = i - 385
            current_time += time_incr / fs
    for i in range(129):
        if instruments[i].notes:
            midi.instruments.append(instruments[i])
    return midi

class Dataset(data.Dataset):
    def __init__(self, train, length=None):
        super(Dataset, self).__init__()
        if train:
            self.pathlist = trainlist
        else:
            self.pathlist = testlist
        self.len = length

    def __getitem__(self, index):
        while True:
            try:
                path = self.pathlist[np.random.randint(0, len(self.pathlist))]
                return midi_roll(path)
            except:
                # tqdm.write(path)
                continue

    def __len__(self):
        if self.len is None:
            return len(self.pathlist)
        return self.len

def init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=16, train=True, length=None):
        super(DataLoader, self).__init__(Dataset(train, length), \
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
    #song = midi_roll('Datasets/lmd_matched/A/X/L/TRAXLZU12903D05F94/1bddc5dbd78f2d242a02a9985bc6b400.mid')
    #midi = piano_rolls_to_midi(song)
    #midi.write('Samples/Never.mid')
    # time_list = []
    # for _ in tqdm(range(100)):
    #     while True:
    #         try:
    #             time_list_datum = midi_roll(pathlist[np.random.randint(0, len(pathlist))])
    #             break
    #         except:
    #             continue
    #     time_list += time_list_datum
    # time_list.sort()
    # time_list = time_list[:-len(time_list) // 100]
    # print(min(time_list))
    # print(max(time_list))
    # bins = np.exp(np.arange(np.log(min(time_list)), np.log(max(time_list)), (np.log(max(time_list)) - np.log(min(time_list))) / 1000))
    # plt.xscale('log', nonposx='clip')
    # plt.hist(time_list, bins=bins, cumulative=True, histtype='step')
    # plt.show()
    # plt.close()
    # time_list = []
    # sort_list = []
    len_list = []
    for _ in tqdm(range(1000)):
        while True:
            try:
                length = test_function(random.choice(pathlist))
                # sort_list_frag, time_list_frag = test_function(random.choice(pathlist))
                break
            except:
                continue
        len_list.append(length)
    len_list.sort()
    print(len_list)
    # plt.hist([leng[0] for leng in len_list])
    # plt.show()
    # plt.show()
        # sort_list += sort_list_frag
        # time_list += time_list_frag
        # bins = np.logspace(np.log(min(sort_list_frag)), np.log(max(sort_list_frag)), num=100, base=np.e)
        # plt.xscale('log', nonposx='clip')
        # plt.hist(sort_list_frag, bins=bins, cumulative=True, histtype='step')
        # plt.show()
        # plt.close()
        # time_list_frag = [time for time in time_list_frag if time != 0]
        # bins = np.logspace(np.log(min(time_list_frag)), np.log(max(time_list_frag)), num=100, base=np.e)
        # plt.xscale('log', nonposx='clip')
        # plt.hist(time_list_frag, bins=bins, cumulative=True, histtype='step')
        # plt.show()
        # plt.close()
    

if __name__ == '__main__':
    Test()
