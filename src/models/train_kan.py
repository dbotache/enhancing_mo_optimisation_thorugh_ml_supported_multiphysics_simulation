from models.kan import KAN, train_epoch, valid_epoch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import ParameterData
import pandas as pd


def train_model(model_parameters, epochs, batch_size, X_train, y_train,
                X_test, y_test, criterion, device, n_cpu=1, verbose=False):

    train_dataset = ParameterData(X_train, y_train)
    test_dataset = ParameterData(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_cpu)

    lr = model_parameters['lr']
    weight_decay = model_parameters['weight_decay']
    hidden_size = model_parameters['hidden_size']
    n_layers = model_parameters['n_layers']
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    layers_hidden = [input_size] + [hidden_size for i in range(n_layers)] + [output_size]

    param_kan = {
        'layers_hidden': layers_hidden,
        'grid_size': model_parameters['grid_size'],
        'spline_order': model_parameters['spline_order'],
        'scale_noise': model_parameters['scale_noise'],
        'grid_eps': model_parameters['grid_eps']
    }

    model = KAN(**param_kan)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)

        test_loss = valid_epoch(model, device, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        test_loss = test_loss / len(test_loader.sampler)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if verbose:
            print(f'Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss} | Test Loss: {test_loss}')

    return model, pd.DataFrame(history)

