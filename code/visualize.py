import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, logits, image_labels, labels):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 3, 32, 32)
    :param logits: the output of model(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50,)
    :param labels: list of class names, e.g., ["cat", "deer", "dog"]
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    if isinstance(image_inputs, torch.Tensor):
        image_inputs = image_inputs.numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    if isinstance(image_labels, torch.Tensor):
        image_labels = image_labels.numpy()

    image_inputs = np.transpose(image_inputs, (0, 2, 3, 1))

    predicted_labels = np.argmax(logits, axis=1)
    num_images = image_inputs.shape[0]

    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind])
            pl = labels[predicted_labels[ind]]
            al = labels[image_labels[ind]]
            ax.set(title=f"PL: {pl}\nAL: {al}")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == image_labels[i]:
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()