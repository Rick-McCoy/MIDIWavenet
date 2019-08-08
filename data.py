import warnings
import pathlib
from torch.utils import data
import pretty_midi as pm
import numpy as np
from utils import init_fn

PATHLIST = list(pathlib.Path('Datasets').glob('**/*.[Mm][Ii][Dd]'))
# with open('lmd_pathlist.txt', 'r') as f:
#     PATHLIST = f.readlines()
# PATHLIST = [x.strip() for x in PATHLIST]
# with open('pathlist.txt', 'r') as f:
#     ADD_PATHLIST = f.readlines()
# PATHLIST += [x.strip() for x in ADD_PATHLIST]
np.random.shuffle(PATHLIST)
TRAIN_LIST = PATHLIST[:-1024]
TEST_LIST = PATHLIST[-1024:]

def midi_roll(path, input_length, output_length):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path).replace('\\', '/'))
    event_list = []
    global_condition = np.zeros((256), dtype=np.float32)
    for inst in song.instruments:
        if inst.is_drum:
            for note in inst.notes:
                if note.velocity:
                    event_list.append((
                        int(note.start * 32768), 6,
                        note.pitch,
                        note.pitch,
                        note.velocity
                    ))
                    event_list.append((
                        int(note.end * 32768), 7,
                        note.pitch,
                        note.pitch,
                        0
                    ))
                    global_condition[128 + note.pitch] = 1
        else:
            global_condition[inst.program] = 1
            for note in inst.notes:
                if note.velocity:
                    event_list.append((
                        int(note.start * 32768), 3,
                        inst.program,
                        note.pitch,
                        note.velocity
                    ))
                    event_list.append((
                        int(note.end * 32768), 4,
                        inst.program,
                        note.pitch,
                        0
                    ))
    event_list.sort()
    input_list = [1] * (input_length // 4 * 4)
    current_time = event_list[0][0]
    for event in event_list:
        if event[0] > current_time:
            time = min(event[0] - current_time, 32767)
            input_list += [5, time % 32, time // 32 % 32, time // 1024]
            current_time = event[0]
        input_list += event[1:]
    input_list += [2] * (output_length // 4 * 4)
    num = np.random.randint(0, len(input_list) - input_length + 1) // 4 * 4
    target = np.array(input_list[num : num + input_length], dtype=np.int64)
    local_condition = [1, 2, 3, 4] * (input_length // 4 * 4) + [1, 2, 3, 4][:input_length % 4]
    local_condition = np.array(local_condition, dtype=np.float32)
    return target, global_condition#, local_condition

def piano_rolls_to_midi(roll):
    midi = pm.PrettyMIDI(resolution=960)
    instruments = [pm.Instrument(i) for i in range(128)] \
                + [pm.Instrument(0, is_drum=True)]
    current_time = 0
    start_time = [[[]] * 128] * 129
    roll = [roll[i : i + 4] for i in range(0, len(roll), 4)]
    for event in roll:
        if event[0] == 1:
            continue
        elif event[0] == 2:
            break
        elif event[0] == 3 or event[0] == 6:
            instrument = 128 if event[0] == 6 else event[1]
            start_time[instrument][event[2]].append((current_time, event[3]))
        elif event[0] == 4 or event[0] == 7:
            instrument = 128 if event[0] == 7 else event[1]
            for start, velocity in start_time[instrument][event[2]]:
                if current_time > start:
                    instruments[instrument].notes.append(
                        pm.Note(
                            velocity=velocity,
                            pitch=event[2],
                            start=start / 32768,
                            end=current_time / 32768
                        )
                    )
            start_time[instrument][event[2]] = []
        elif event[0] == 5:
            current_time = current_time + (event[1] + event[2] * 32 + event[3] * 1024)
    for inst in instruments:
        if inst.notes:
            midi.instruments.append(inst)
    return midi

class Dataset(data.Dataset):
    def __init__(self, train, input_length, output_length, dataset_length):
        super(Dataset, self).__init__()
        self.pathlist = np.array(TRAIN_LIST if train else TEST_LIST)
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_length = dataset_length

    def __getitem__(self, index):
        while True:
            try:
                return midi_roll(
                    np.random.choice(self.pathlist),
                    self.input_length,
                    self.output_length
                )
            except IndexError:
                continue

    def __len__(self):
        return self.dataset_length if self.dataset_length else self.pathlist.shape[0]

class DataLoader(data.DataLoader):
    def __init__(
            self,
            batch_size,
            shuffle=True,
            num_workers=16,
            train=True,
            input_length=0,
            output_length=0,
            dataset_length=0
    ):
        super(DataLoader, self).__init__(
            Dataset(
                train,
                input_length,
                output_length,
                dataset_length
            ),
            batch_size,
            shuffle,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=init_fn
        )
