import pickle
import numpy as np
import torch


def unpickle(file):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.
    NOTE: DO NOT EDIT
    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_next_batch(idx, inputs, labels, batch_size=100):
    """
    Given an index, returns the next batch of data and labels. Ex. if batch_size is 5, 
    the data will be a tensor of size 5 * 3 * 32 * 32, and the labels returned will be a tensor of size 5.
    """
    batch_inputs = inputs[idx*batch_size:(idx+1)*batch_size]
    batch_labels = labels[idx*batch_size:(idx+1)*batch_size]
    return batch_inputs, batch_labels


def get_data(file_path, classes):
    """
    Given a file path and a list of class indices, returns an array of 
    normalized inputs (images) and an array of labels. 
    
    - You will want to first extract only the data that matches the 
    corresponding classes we want.
    - You should make sure to normalize all inputs and also return the labels
    as class indices.
    
    :param file_path: file path for inputs and labels, something 
                        like 'CIFAR_data_compressed/train'
    :param classes: list of class labels (0-9) to include in the dataset

    :return: normalized PyTorch tensor of inputs and tensor of labels, where 
                inputs are of type torch.float32 and has size (num_inputs, num_channels, height, width) 
                and labels tensor with size (num_examples,)
    """
    unpickled_file: dict[str, np.ndarray] = unpickle(file_path)
    inputs: np.ndarray = np.array(unpickled_file[b'data'])
    labels: np.ndarray = np.array(unpickled_file[b'labels'])

    # TODO: Extract only the data that matches the corresponding classes we want
    raise NotImplementedError("Implement me!")


if __name__ == "__main__":
    get_data("./data/train", [0, 1])