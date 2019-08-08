# MIDIWavenet

Modified Wavenet that works on MIDI files.

## Getting Started

### Prerequisites

```
pytorch>=1.1.0
tqdm
numpy
pretty-midi
matplotlib
tb-nightly>=1.14.0
```

### Installing

CUDA & cudnn is neccessary. Use whatever version that suits your libraries.

Assumes training uses GPUs, will throw error if no CUDA-capable GPUs are present.

Have tested on 1-GPU & 4-GPU environments.

### Datasets

Training data that I used are from two sources:

[The Lakh MIDI Dataset v1.1](https://colinraffel.com/projects/lmd/): LMD_matched was used.

[The Largest MIDI Collection on the Internet](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/): A small subset was used.

Extract additional datasets under /Datasets.

For fast preparation of filenames, all filenames are indexed in a text file under /Datasets. Add additional files if neccessary.

### Training

Run

```
python3 train.py
```

All hyperparameters are modifiable via flags. Please refer to train.py for specifics.

### Sampling

Make sure your checkpoint is under /Checkpoints.

Then, run

```
python3 train.py --sample NUMBER_OF_SAMPLES --resume RESUME_CHECKPOINT_NUMBER
```

If no --resume flag is given, the checkpoint with the largest number in its name will be selected.

Generated samples & piano roll image files will be under /Samples.

## TODOs

~~Enable general generation of MIDI files: Currently only able to generate classical music.~~
    -> General generation is now possible: only need to expand dataset. Unfortunately, specifying genre is currently impossible.

~~Add length flag for sampling: Currently fixed at 4096 time steps. (\~43 seconds)~~
    -> ~~Added length flag. Unit of length: 1/96 (s)~~.
    -> Currently, MIDIWavenet decided when to end. Maxes out in 10000 time steps, or \~100 seconds.

Improve quality of generated music.

## Authors

* **Rick-McCoy** - *Initial work* - [Rick-McCoy](https://github.com/Rick-McCoy)

## License

This project is licensed under the MIT License.

## Acknowledgments

* **Jin-Jung Kim** - *General structure of code adapted* - [golbin](https://github.com/golbin)
* **Vincent Hermann** - *pytorch-wavenet repository was a big help* - [vincenthermann](https://github.com/vincentherrmann)
* **Everyone Else I Questioned** - *Thanks!*
