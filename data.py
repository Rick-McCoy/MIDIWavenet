import os
import pathlib
from tqdm import tqdm
import pretty_midi as pm
import numpy as np
import torch
import warnings
import re
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data

INPUT_LENGTH = 8192
MAX_LENGTH = 32768
pathlist = list(pathlib.Path('Datasets/Classics').glob('**/*.mid'))
trainlist = pathlist[:-144]
testlist = pathlist[-144:]

def natural_sort_key(s, _nsre=re.compile('(\\d+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def full_piano_roll(path, receptive_field):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(midi_file=str(path))
    piano_rolls = [(_.get_piano_roll(fs=song.resolution), _.program) for _ in song.instruments if not _.is_drum]
    drum_rolls = [(_.get_piano_roll(fs=song.resolution), _.program) for _ in song.instruments if _.is_drum]
    length = np.amax([roll.shape[1] for roll, _ in piano_rolls + drum_rolls])
    data = np.zeros(shape=(128 * 129 + 2, length))
    condition = np.zeros(shape=(129, 1))
    for roll, instrument in piano_rolls:
        data[instrument * 128: (instrument + 1) * 128] += np.pad(roll, [(0, 0), (0, length - roll.shape[1])], 'constant')
        condition[instrument] = 1
    for roll, instrument in drum_rolls:
        data[128 * 128 : 128 * 129] += np.pad(roll, [(0, 0), (0, length - roll.shape[1])], 'constant')
        condition[-1] = 1
    num = np.random.randint(0, length)
    data = data[:, num:num + MAX_LENGTH]
    data[-2] += 1 - data[:-2].sum(axis=0)
    length = data.shape[1]
    if length < MAX_LENGTH:
        data = np.pad(data, [(0, 0), (0, MAX_LENGTH - length)], 'constant')
        data[-1, length - MAX_LENGTH] = 1
    data = data > 0
    answer = np.transpose(data[:, receptive_field + 1:], (1, 0))
    diff = np.sum(np.diff(data)[:, receptive_field:], axis=0, keepdims=True) > 0
    return data.astype(np.float32), answer.astype(np.float32), diff.astype(np.float32), condition.astype(np.float32)

def piano_roll(path, receptive_field):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(midi_file=str(path))
    classes = [0, 3, 5, 7, 8, 9]
    limits = [[24, 96], [36, 84], [24, 96], [36, 84], [36, 84], [60, 96]]
    limit_slice = [0, 72, 120, 192, 240, 288, 324]
    piano_rolls = [(_.get_piano_roll(fs=song.resolution), _.program) for _ in song.instruments if not _.is_drum and _.program // 8 in classes]
    length = np.amax([roll.shape[1] for roll, _ in piano_rolls])
    data_full = np.zeros(shape=(326, length))
    condition = np.zeros(shape=(6, 1))
    for roll, instrument in piano_rolls:
        i = classes.index(instrument // 8)
        sliced_roll = roll[limits[i][0]:limits[i][1]]
        data_full[limit_slice[i]:limit_slice[i + 1]] += np.pad(sliced_roll, [(0, 0), (0, length - sliced_roll.shape[1])], 'constant')
        condition[i] = 1
    num = np.random.randint(0, length)
    data = data_full[:, num : INPUT_LENGTH + num]
    data[324] += 1 - data[:324].sum(axis = 0)
    length = data.shape[1]
    if length < INPUT_LENGTH:
        data = np.pad(data, [(0, 0), (0, INPUT_LENGTH - length)], 'constant')
        data[325, length - INPUT_LENGTH:] = 1
    data = data > 0
    answer = np.transpose(data[:, receptive_field + 1:], axes=(1, 0))
    diff = np.sum(np.diff(data)[:, receptive_field:], axis=0, keepdims=True) > 0
    return data.astype(np.float32), answer.astype(np.float32), diff.astype(np.float32), condition.astype(np.float32)

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
    def __init__(self, train, receptive_field):
        super(Dataset, self).__init__()
        if train:
            self.pathlist = trainlist
        else:
            self.pathlist = testlist
        self.receptive_field = receptive_field
    
    def __getitem__(self, index):
        data = piano_roll(self.pathlist[index], self.receptive_field)
        return data

    def __len__(self):
        return len(self.pathlist)

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, receptive_field, shuffle=True, num_workers=16, train=True):
        super(DataLoader, self).__init__(Dataset(train, receptive_field), batch_size, shuffle, num_workers=num_workers)

def Test():
    pathlist = list(pathlib.Path('Datasets/Classics').glob('**/*.mid'))
    for path in tqdm(pathlist[:1]):
        data, answer, diff, condition = piano_roll(path, 5115)
        tqdm.write(str(data.shape))
        tqdm.write(str(answer.shape))
        tqdm.write(str(diff.shape))
        tqdm.write(str(condition.shape))

if __name__ == '__main__':
    Test()
