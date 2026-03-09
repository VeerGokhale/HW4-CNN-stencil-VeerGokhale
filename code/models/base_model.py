from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class CifarModel(nn.Module, ABC):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.loss_list = []
        # TODO: Initialize loss function
        self.loss_func = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, inputs):
        """
        No need to implement this. You will override this in your child classes!
        """
        pass

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        :param labels: during training, tensor of shape (batch_size,) containing class indices
        :return: the loss of the model as a Tensor
        """
        # TODO: Implement the loss function

        loss = self.loss_func(logits, labels)

        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        :param logits: a matrix of size (num_inputs, self.num_classes)
        :param labels: tensor of size (num_labels,) containing class indices
        :return: the accuracy of the model as a Tensor
        """
        preds = torch.argmax(logits, dim=1)
        correct = preds == labels
        acc = torch.mean(correct.to(torch.float32))
        return acc
