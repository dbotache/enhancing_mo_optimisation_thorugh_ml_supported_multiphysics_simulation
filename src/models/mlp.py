import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP is a feedforward neural network architecture that consists of a stack of fully connected layers with ReLU
    activation functions.

    Args:
        input_size (int): The size of the input tensor. It should be a 1D tensor.
        hidden_size (int): The number of units in each hidden layer.
        n_layers (int): The number of hidden layers in the network.
        output_size (int): The number of output classes.
        dropout_prob (float): The probability of dropout for the network. Default is 0.0.
        act (str): Activation function for the hidden layers. Can be 'relu' or 'elu'. Default is 'relu'.
        alpha (float): Alpha value for the ELU activation function. Default is 1.0.

    Returns:
        A tensor representing the output of the network.
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout_prob=.0, activation='relu', alpha=1.0):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            elif i == (n_layers - 1):
                self.layers.append(nn.Linear(hidden_size, output_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout_prob)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=alpha)
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    model.train()

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        output = model(X_batch.float())
        loss = loss_fn(output, y_batch.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss = 0.0
    model.eval()

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch.float())
        loss = loss_fn(output, y_batch.float())
        valid_loss += loss.item()

    return valid_loss
