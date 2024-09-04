import torch
import torch.nn as nn


class SingleShotCNN(nn.Module):
    """
    SingleshotCNN is a convolutional neural network (CNN) architecture that consists of a stack of convolutional layers
    followed by a fully connected layer.

    Args:
        input_size (int): The size of the input tensor. It should be a 1D tensor.
        output_size (int): The number of targets.
        hidden_channel_size (int): The number of output channels in each convolutional layer. Default is 50.
        n_layers (int): The number of convolutional layers in the network. Default is 3.
        kernel_size (int): The size of the convolutional kernel. Default is 3.
        stride (int): The stride of the convolutional kernel. Default is 1.
        padding (int): The padding size for the convolutional layers. Default is 0.
        dropout_prob (float): The probability of dropout for the network. Default is 0.0.
        act (str): Activation function for the hidden layers. Can be 'relu' or 'elu'. Default is 'relu'.
        alpha (float): Alpha value for the ELU activation function. Default is 1.0.

    Returns:
        A tensor representing the output of the network.
    """

    def __init__(self, input_size, output_size, hidden_channel_size=50, n_layers=3, kernel_size=3, stride=1,
                 padding=0, dropout_prob=.0, activation='relu', alpha=1.0):
        super(SingleShotCNN, self).__init__()

        self.output_size = output_size

        hidden_channels = [hidden_channel_size for i in range(n_layers)]
        sizes = [1] + hidden_channels

        # Define the convolutional layers
        self.cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=sizes[i],
                                                   out_channels=sizes[i + 1],
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding) for i in range(len(sizes) - 1)])

        input_size_ = input_size
        for layer in self.cnn_layers:
            input_size_ = int((input_size_ - kernel_size + 2 * padding) / stride) + 1

        fc_input_size = input_size_ * sizes[-1]

        # Define the fully connected layer
        self.fc = nn.Linear(in_features=fc_input_size, out_features=self.output_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=alpha)

    def forward(self, x):
        # Pass the input through the convolutional layers
        for layer in self.cnn_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)

        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        # Pass the flattened output through the fully connected layers
        x = self.fc(x)

        return x


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    model.train()

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        output = model(X_batch.float())
        loss = loss_fn(output, y_batch.view(-1, y_batch.shape[2]).float())
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
        loss = loss_fn(output, y_batch.view(-1, y_batch.shape[2]).float())
        valid_loss += loss.item()

    return valid_loss
