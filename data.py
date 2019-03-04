import os
import pathlib
from tqdm import tqdm
import pretty_midi as pm
import numpy as np
import torch
import warnings
import re
import librosa.display
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data

INPUT_LENGTH = 4096
NON_LENGTH = 512
pathlist = list(pathlib.Path('Datasets/Classics').glob('**/*.mid')) + list(pathlib.Path('Datasets/Classics').glob('**/*.MID'))
np.random.shuffle(pathlist)
trainlist = pathlist[:-768]
testlist = pathlist[-768:]

def natural_sort_key(s, _nsre=re.compile('(\\d+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def piano_roll(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(midi_file=str(path))
    classes = [0, 3, 5, 7, 8, 9]
    limits = [[24, 96], [36, 84], [24, 96], [36, 84], [36, 84], [60, 96]]
    limit_slice = [0, 72, 120, 192, 240, 288, 324]
    piano_rolls = [(_.get_piano_roll(fs=song.resolution), classes.index(_.program // 8)) for _ in song.instruments if not _.is_drum and _.program // 8 in classes]
    length = np.amax([roll.shape[1] for roll, _ in piano_rolls])
    data_full = np.zeros(shape=(326, length))
    condition = np.zeros(shape=(6))
    shift = np.random.randint(-2, 3)
    for roll, i in piano_rolls:
        sliced_roll = roll[limits[i][0] + shift:limits[i][1] + shift]
        data_full[limit_slice[i]:limit_slice[i + 1], -sliced_roll.shape[1]:] += sliced_roll
        condition[i] = 1
    if length < INPUT_LENGTH:
        data_full = np.pad(data_full, [(0, 0), (INPUT_LENGTH - length, 0)], 'constant')
        data_full[-1, :INPUT_LENGTH - length] = 1
        length = INPUT_LENGTH
    num = np.random.randint(0, length - INPUT_LENGTH + 1)
    data = data_full[:, num : INPUT_LENGTH + num]
    data[324] = data[:324].sum(axis = 0) == 0
    data = data != 0
    diff = np.zeros(shape=(7, INPUT_LENGTH - 1))
    data_diff = np.diff(data) != 0
    for i in range(6):
        diff[i] += data_diff[limit_slice[i]:limit_slice[i + 1]].sum(axis=0)
    diff[-1] = diff[:-1].sum(axis=0) == 0
    diff = np.ascontiguousarray(diff.transpose() != 0)
    indices = (np.diff(data_full) != 0).sum(axis=0).nonzero()[0]
    nonzero = data_full[:, indices + 1]
    length = nonzero.shape[1]
    if length < NON_LENGTH:
        nonzero = np.pad(nonzero, [(0, 0), (NON_LENGTH - length, 0)], 'constant')
        nonzero[-1, :NON_LENGTH - length] = 1
        length = NON_LENGTH
    num = np.random.randint(0, length - NON_LENGTH + 1)
    nonzero = nonzero[:, num : NON_LENGTH + num]
    nonzero[324] = nonzero[:324].sum(axis=0) == 0
    nonzero = nonzero != 0
    nonzero_diff = np.zeros(shape=(7, NON_LENGTH - 1))
    nonzero_diff_bin = np.diff(nonzero) != 0
    for i in range(6):
        nonzero_diff[i] += nonzero_diff_bin[limit_slice[i]:limit_slice[i + 1]].sum(axis=0)
    nonzero_diff[-1] = nonzero_diff[:-1].sum(axis=0) != 0
    nonzero_diff = np.ascontiguousarray(nonzero_diff.transpose() != 0)
    return data.astype(np.float32), \
            nonzero.astype(np.float32), \
            diff.astype(np.float32), \
            nonzero_diff.astype(np.float32), \
            condition.astype(np.float32)

def clean(x):
    return x[:-2]

def save_roll(x, step):
    fig = plt.figure(figsize=(72, 24))
    librosa.display.specshow(x, x_axis='time', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
    plt.title('{}'.format(step))
    fig.savefig('Samples/{}.png'.format(step))
    plt.close(fig)

def piano_rolls_to_midi(x, fs=96):
    channels = [72, 48, 72, 48, 48, 36]
    for i in range(1, 6):
        channels[i] += channels[i - 1]
    x = np.split(x * 100, channels)
    midi = pm.PrettyMIDI()
    limits = [[24, 96], [36, 84], [24, 96], [36, 84], [36, 84], [60, 96]]
    instruments = [0, 24, 40, 56, 64, 72]
    for roll, instrument, limit in zip(x, instruments, limits):
        current_inst = pm.Instrument(instrument)
        current_roll = np.pad(roll, [(limit[0], 128 -  limit[1]), (1, 1)], 'constant')
        notes = current_roll.shape[0]
        velocity_changes = np.nonzero(np.diff(current_roll).T)
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)
        for time, note in zip(*velocity_changes):
            velocity = current_roll[note, time + 1]
            time /= fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                if time > note_on_time[note] + 1 / fs:
                    pm_note = pm.Note(
                        velocity=prev_velocities[note], 
                        pitch=note, 
                        start=note_on_time[note], 
                        end=time
                    )
                    current_inst.notes.append(pm_note)
                prev_velocities[note] = 0
        midi.instruments.append(current_inst)
    return midi

class Dataset(data.Dataset):
    def __init__(self, train):
        super(Dataset, self).__init__()
        if train:
            self.pathlist = trainlist
        else:
            self.pathlist = testlist

    def __getitem__(self, index):
        return piano_roll(self.pathlist[index])

    def __len__(self):
        return len(self.pathlist)

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=16, train=True):
        super(DataLoader, self).__init__(Dataset(train), batch_size, shuffle, num_workers=num_workers)

def Test():
    timelist_1 = []
    timelist_2 = []
    for path in tqdm(pathlist[:100]):
        *_, time_1, time_2, time_3 = piano_roll(str(path))
        if time_2 < time_1:
            timelist_1.append(time_2 / time_1)
            timelist_2.append(time_3 / time_1)
    plt.scatter(timelist_1, timelist_2)
    # plt.hist(timelist_1, bins=100)
    plt.show()
    plt.close()
    # plt.hist(timelist_2, bins=100)
    # plt.show()
    # plt.close()

if __name__ == '__main__':
    Test()
