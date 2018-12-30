# MIDIWavenet

Modified Wavenet that works on MIDI files.

## Getting Started

### Prerequisites

```
pytorch
tqdm
numpy
pretty-midi
matplotlib
```

### Installing

None necessary. Clone this repository.

### Datasets

Training data are from two sources:

[The Lakh MIDI Dataset v1.1](https://colinraffel.com/projects/lmd/): LMD_matched was used.

[The Largest MIDI Collection on the Internet](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/): A small subset was used.

Extract datasets under /Datasets/Classics.

### Training

Run

```
python3 train.py
```

All hyperparameters are modifiable via flags. Please refer to train.py for specifics.

### Sampling

Make sure your checkpoints are under /Checkpoints.

The two checkpoints with the largest number in its name will be automatically selected.

Then, run

```
python3 train.py --sample NUMBER_OF_SAMPLES
```

## TODOs

Enable general generation of MIDI files: Currently only able to generate classical music.

~~Add length flag for sampling: Currently fixed at 4096 time steps. (\~43 seconds)~~ Added length flag. Unit of length: 1/96 (s).

## Authors

* **Rick-McCoy** - *Initial work* - [Rick-McCoy](https://github.com/Rick-McCoy)

## License

This project is licensed under the MIT License.

## Acknowledgments

* **Jin-Jung Kim** - *General structure of code adapted* - [golbin](https://github.com/golbin)
* **Vincent Hermann** - *pytorch-wavenet repository was a big help* - [vincenthermann](https://github.com/vincentherrmann)
* **Everyone Else I Questioned** - *Thanks!*
