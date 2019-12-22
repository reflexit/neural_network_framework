#!/usr/bin/python3

import argparse

from src.neural_network import neural_network_wrapper


def main():
    """
    Parse arguments, and call corresponding functions.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Neural network framework",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-t", "--task",
                        help="the task to run (default: 0)\n"
                             "0: print model summary; 1: train; 2: test; 3: train + test",
                        default=0,
                        type=int)
    parser.add_argument("-m", "--model_path",
                        help="the model file path (default: bin/model.h5)",
                        default="bin/model.h5",
                        type=str)

    args = parser.parse_args()
    task = args.task
    model_path = args.model_path

    if task in {0, 1, 2, 3}:
        neural_network_wrapper(task, model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
