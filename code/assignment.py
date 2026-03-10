from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from visualize import visualize_loss, visualize_results
#from models.cnn import CNN
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
    model.train()

    perm = torch.randperm(train_inputs.shape[0])
    train_inputs = train_inputs[perm]
    train_labels = train_labels[perm]

    num_batches = (train_inputs.shape[0] + model.batch_size - 1) // model.batch_size
    total_acc = 0.0

    with tqdm(range(num_batches), desc="  train", unit="batch", leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}") as pbar:
        for batch_idx in pbar:
            batch_inputs, batch_labels = get_next_batch(
                batch_idx, train_inputs, train_labels, model.batch_size
            )

            optimizer.zero_grad()

            logits = model(batch_inputs)
            loss = model.loss(logits, batch_labels)
            batch_acc = model.accuracy(logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_acc += batch_acc.item()

    return total_acc / num_batches


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
    model.eval()

    num_batches = (test_inputs.shape[0] + model.batch_size - 1) // model.batch_size
    total_acc = 0.0
    all_preds = []

    with torch.no_grad():
        with tqdm(range(num_batches), desc="  test", unit="batch", leave=False,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}") as pbar:
            for batch_idx in pbar:
                batch_inputs, batch_labels = get_next_batch(
                    batch_idx, test_inputs, test_labels, model.batch_size
                )

                logits = model(batch_inputs)
                batch_acc = model.accuracy(logits, batch_labels)

                total_acc += batch_acc.item()
                all_preds.append(logits)


    test_accuracy = total_acc / num_batches
    test_preds = torch.cat(all_preds, dim=0)

    return test_accuracy, test_preds

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
    x_train, y_train = get_data(LOCAL_TRAIN_FILE, class_idxs)
    x_test, y_test = get_data(LOCAL_TEST_FILE, class_idxs)

    # Initialize model
    # TODO: main() pt 2
    # Initialize your model and optimizer

    model = MLP(class_idxs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    # TODO: main() pt 3
    # Train your model for multiple epochs

    num_epochs = 10
    for epoch in range(num_epochs):
        train_acc = train(model, optimizer, x_train, y_train)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_acc:.4f}")
        
    # TODO: main() pt 4
    # Test your model

    test_acc, test_preds = test(model, x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # TODO: main() pt 5
    # Save your predictions as either "predictions_cnn.npy" or "predictions_mlp.npy"
    #   depending on which model you are using

    pred_classes = torch.argmax(test_preds, dim=1)
    np.save("predictions_mlp.npy", pred_classes.cpu().numpy())





if __name__ == '__main__':
    main()