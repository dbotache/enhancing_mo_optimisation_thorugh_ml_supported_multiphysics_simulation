import os
import json
import numpy as np
import ax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import ParameterData
from models.cnn import SingleShotCNN
from models.train_cnn import train_model
from sklearn.model_selection import KFold
from ressources import n_cpu


SEARCH_SPACE = [{'name': 'lr', 'type': 'choice',
                 'values': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                 'value_type' : 'float', 'is_ordered' : True},
                {'name': 'dropout_prob', 'type': 'choice',
                 'values': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 'value_type' : 'float', 'is_ordered' : True},
                {'name': 'weight_decay', 'type': 'choice',
                 'values': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 'value_type' : 'float', 'is_ordered' : True},
                {'name': 'hidden_channel_size', 'type': 'choice',
                 'values': [25, 50, 100, 150, 200, 250, 300, 350, 400],
                 'value_type' : 'int', 'is_ordered' : True},
                {'name': 'n_layers', 'type': 'range', 'bounds': [2, 8], 'value_type' : 'int'},
                {'name': 'kernel_size', 'type': 'choice',
                 'values': [3, 5, 7],
                 'value_type' : 'int', 'is_ordered' : True},
                {'name': 'padding', 'type': 'range', 'bounds': [1, 2], 'value_type' : 'int'},
                {'name': 'activation', 'type': 'choice', 'values': ['relu', 'elu'], 'value_type': 'str'}]

MODEL_PARAMETERS = {"lr": 0.001,
                    "dropout_prob": 0.01,
                    "weight_decay": 0.05,
                    "hidden_channel_size": 100,
                    "n_layers": 2,
                    "kernel_size": 5,
                    "padding": 2,
                    "activation": "elu",
                    "feature_scaler": "minmax",
                    "target_scaler": "minmax"}


def CrossValTrain(parameters, num_epochs, batch_size, X_train, y_train, device, n_splits=5, metric='MSE'):
    """
    Train a Single Shot CNN using k-fold cross-validation and return the mean loss.

    Args:

    parameters (dict): dictionary of hyperparameters for SingleShotCNN model and optimizer
    num_epochs (int): number of epochs to train the model
    batch_size (int): batch size for training and validation DataLoader
    X_train (pandas DataFrame): training input data
    y_train (pandas DataFrame): training target data
    device (str): device to use for training ('cpu' or 'cuda')
    n_splits (int, optional): number of splits for k-fold cross-validation (default=5)
    Returns:

    mean_loss (float): mean validation loss across all k-folds
    Note:

    This function uses the KFold class from sklearn.model_selection to split the data into k-folds.
    The data is wrapped in TensorDataset and DataLoader for efficient training.
    The loss function used is nn.MSELoss with reduction='sum'.
    The optimizer used is optim.SGD with learning rate and weight decay specified in parameters dict.
    The best loss for each fold is stored in a list and the mean loss is returned as the objective to be optimized by Ax.
    """

    # Define the k-fold cross validation
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # Initialize the loss array
    cv_loss = []

    i = 0
    for train_index, val_index in kfold.split(X_train):

        i += 1
        # print(f"evaluating split {i}")

        X_train_, X_val_ = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[val_index]

        # Wrap the data in TensorDataset and DataLoader for efficient training
        train_dataset = ParameterData(X_train_, y_train_)
        val_dataset = ParameterData(X_val_, y_val_)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

        # Initialize the model, loss function, and optimizer
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        hidden_channel_size = parameters['hidden_channel_size']
        n_layers = parameters['n_layers']
        kernel_size = parameters['kernel_size']
        padding = parameters['padding']
        dropout_prob = parameters['dropout_prob']

        try:
            activation = parameters['activation']
        except:
            activation = 'relu'
        try:
            alpha = parameters['alpha']
        except:
            alpha = 1.0

        model = SingleShotCNN(input_size=input_size,
                              output_size=output_size,
                              hidden_channel_size=hidden_channel_size,
                              n_layers=n_layers,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              dropout_prob=dropout_prob,
                              activation=activation,
                              alpha=alpha)

        model.to(device)

        if metric == 'MSE':
            criterion = nn.MSELoss(reduction='sum')
        elif metric == 'MAE':
            criterion = nn.L1Loss(reduction='sum')

        optimizer = optim.Adam(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])

        # Train the model
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.view(-1, output_size).float())
                loss.backward()
                optimizer.step()

                # print(loss)

        # Evaluate the model on the validation set
        with torch.no_grad():
            total_loss = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.view(-1, output_size).float())
                total_loss += loss.item() * len(labels)
                total += len(labels)
            mean_loss = total_loss / total

        # Store the best loss for each fold
        cv_loss.append(mean_loss)

    # Return the mean loss as the objective to be optimized by Ax
    print(f'Loss: {np.mean(cv_loss)} and parameters: {parameters}')
    return np.mean(cv_loss)


def optimize_cnn(args, X_train, y_train, X_test, y_test, device, cv_n_splits=5, n_cpu=1, trials=10, metric='MSE'):
    num_epochs = args.model.num_epochs
    batch_size = args.model.batch_size

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name)):
        os.makedirs(os.path.join(args.main_path, "models", args.model.model_type, args.file_name))

    # Define the objective function to be minimized
    def objective(parameters):
        return CrossValTrain(parameters, num_epochs, batch_size, X_train, y_train, device,
                             n_splits=cv_n_splits, metric=metric)

    # Optimize the hyperparameters using Ax
    best_parameters, values, experiment, model = ax.optimize(
        parameters=SEARCH_SPACE,
        evaluation_function=objective,
        total_trials=trials,
        minimize=True
    )

    best_parameters['feature_scaler'] = args.model.feature_scaler
    best_parameters['target_scaler'] = args.model.target_scaler

    print("Best hyperparameters: \n", best_parameters)

    with open(os.path.join(args.main_path, "models", args.model.model_type,
                           args.file_name, 'model_parameters.json'), 'w') as f:
        json.dump(best_parameters, f)

    if metric == 'MSE':
        criterion = nn.MSELoss(reduction='sum')
    elif metric == 'MAE':
        criterion = nn.L1Loss(reduction='sum')

    model, history = train_model(best_parameters, num_epochs, batch_size, X_train, y_train,
                X_test, y_test, criterion, device, n_cpu=n_cpu)

    torch.save(model, f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/model.pt')
    history.to_hdf(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/loss_values.h5', key='loss')


def train_cnn(args, X_train, y_train, X_test, y_test, device, n_cpu=1, metric='MSE'):

    num_epochs = args.model.num_epochs
    batch_size = args.model.batch_size

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name)):
        os.makedirs(os.path.join(args.main_path, "models", args.model.model_type, args.file_name))

    MODEL_PARAMETERS['feature_scaler'] = args.model.feature_scaler
    MODEL_PARAMETERS['target_scaler'] = args.model.target_scaler

    print("Selected hyperparameters: \n", MODEL_PARAMETERS)

    with open(os.path.join(args.main_path,"models", args.model.model_type,
                           args.file_name, 'model_parameters.json'), 'w') as f:
        json.dump(MODEL_PARAMETERS, f)

    if metric == 'MSE':
        criterion = nn.MSELoss(reduction='sum')
    elif metric == 'MAE':
        criterion = nn.L1Loss(reduction='sum')

    model, history = train_model(MODEL_PARAMETERS, num_epochs, batch_size, X_train, y_train,
                X_test, y_test, criterion, device, n_cpu=n_cpu, verbose=True)

    torch.save(model, f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/model.pt')
    history.to_hdf(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/loss_values.h5', key='loss')