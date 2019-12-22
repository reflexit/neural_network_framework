# Neural network framework

This is a general neural network training &amp; testing framework that I use in personal and course projects.
It uses Keras on top of TensorFlow.
You may use this framework subject to the license.

**Note:** This repo is not actively maintained. It may or may not be updated in the future.

## Prerequisites

Step 1: Install **Python 3.6/3.7** and the corresponding **pip**. The framework *may* also run on Python 3.8+.

Step 2: Install requirements: `pip3 install -r resource/requirements.txt`

## How to run

Command to run:
```
python start_scripts.py [-h] [-t TASK] [-m MODEL_PATH]
```

Set `TASK` depending on the specific task to run:
- 0: print model summary
- 1: train
- 2: test
- 3: train + test

Set `MODEL_PATH` as the model file path to save to or load from.

Use `-h` flag to see details about these arguments.

## Changelog

12/22/2019: Opened repo. Finished basic neural network framework.
