from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from visualize import visualize_loss, visualize_results
from models.cnn import CNN
from models.mlp import MLP

import os
import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, optimizer, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels.
    :param model: the initialized model to use for the forward pass and backward pass
    :param optimizer: the optimizer for updating model parameters
    :param train_inputs: train inputs shape (num_inputs, num_channels, height, width)
    :param train_labels: train labels, shape (num_labels,)
    :return: average training accuracy for this epoch
    '''
    num_batches = 10

    with tqdm(range(num_batches), desc="  train", unit="batch", leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}") as pbar:
        for batch_idx in pbar:

            pass

            # Update the progress bar with current loss and accuracy
            # pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

    # TODO: Return average accuracy across all batches
    raise NotImplementedError


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param model: the model to test
    :param test_inputs: test data, shape (num_inputs, num_channels, height, width)
    :param test_labels: test labels, shape (num_labels,)
    :return:
        test accuracy: average accuracy across all batches
        test preds: all of the model's predictions for each of the test inputs
    """
    # NOTE: You might find it helpful to copy some of the printing logic from train()
    # NOTE: Don't forget to put your model in eval mode!
    raise NotImplementedError

def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = './data/train'
    LOCAL_TEST_FILE = './data/test'

    # Load data
    # TODO: main() pt 1
    # Load your testing and training data using the get_data function

    class_idxs = [3, 5]
    x_train, y_train = get_data(AUTOGRADER_TRAIN_FILE, class_idxs)
    x_test, y_test = get_data(AUTOGRADER_TEST_FILE, class_idxs)

    # Initialize model
    # TODO: main() pt 2
    # Initialize your model and optimizer

    model = MLP(class_idxs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    # TODO: main() pt 3
    # Train your model for multiple epochs

    # TODO: main() pt 4
    # Test your model

    # TODO: main() pt 5
    # Save your predictions as either "predictions_cnn.npy" or "predictions_mlp.npy"
    #   depending on which model you are using
    return


if __name__ == '__main__':
    main()