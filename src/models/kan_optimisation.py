import numpy as np
import ax
import os
import json
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import ParameterData
from models.kan import KAN
from models.train_kan import train_model
from sklearn.model_selection import KFold
from ressources import n_cpu
from evaluation.loss_functions import MaskedL1Loss


SEARCH_SPACE = [{'name': 'lr', 'type': 'choice',
                 'values': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                 'value_type' : 'float', 'is_ordered' : True},
                {'name': 'weight_decay', 'type': 'choice',
                 'values': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 'value_type' : 'float', 'is_ordered' : True},
                {'name': 'hidden_size', 'type': 'choice',
                 'values': [25, 50, 100, 200, 400, 800, 1500, 2000, 3000, 3500],
                 'value_type' : 'int', 'is_ordered' : True},
                {'name': 'n_layers', 'type': 'range',
                 'bounds': [2, 16], 'value_type' : 'int'},
                {'name': 'grid_size', 'type': 'range',
                 'bounds': [5, 10], 'value_type': 'int'},
                {'name': 'spline_order', 'type': 'range',
                 'bounds': [2, 3], 'value_type': 'int'},
                {'name': 'scale_noise', 'type': 'range',
                 'bounds': [0.1, 0.5], 'value_type': 'float'},
                {'name': 'grid_eps', 'type': 'range',
                 'bounds': [0.01, 0.1], 'value_type': 'float'},
                ]

MODEL_PARAMETERS = {"lr": 1e-3,
                    "weight_decay": 1e-3,
                    "hidden_size": 50,
                    "n_layers": 2,
                    "grid_size": 5,
                    "spline_order": 3,
                    "scale_noise": .1,
                    "grid_eps": .02
                    }


def CrossValTrain(parameters, num_epochs, batch_size, X_train, y_train, device, n_splits=5, metric='MSE'):
    """
    Train a MLP using k-fold cross-validation and return the mean loss.

    Args:

    parameters (dict): dictionary of hyperparameters for MLP model and optimizer
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

        hidden_size = parameters["hidden_size"]
        n_layers = parameters["n_layers"]
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]

        layers_hidden = [input_size] + [hidden_size for i in range(n_layers)] + [output_size]

        param_kan = {
            'layers_hidden': layers_hidden,
            'grid_size': parameters['grid_size'],
            'spline_order': parameters['spline_order'],
            'scale_noise': parameters['scale_noise'],
            'grid_eps': parameters['grid_eps']
        }

        # Initialize the model, loss function, and optimizer
        model = KAN(**param_kan)
        model.to(device)

        if metric == 'MSE':
            criterion = nn.MSELoss(reduction='sum')
        elif metric == 'MAE':
            criterion = nn.L1Loss(reduction='sum')
        elif metric == "MaskedLoss":
            criterion = MaskedL1Loss(reduction='sum')

        optimizer = optim.Adam(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])

        # Train the model
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.float())
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
                loss = criterion(outputs, labels.float())
                total_loss += loss.item() * len(labels)
                total += len(labels)
            mean_loss = total_loss / total

        # Store the best loss for each fold
        cv_loss.append(mean_loss)

    # Return the mean loss as the objective to be optimized by Ax
    print(f'Loss: {np.mean(cv_loss)} and parameters: {parameters}')
    return np.mean(cv_loss)


def optimize_kan(args, X_train, y_train, X_test, y_test, device, cv_n_splits=5, n_cpu=1, trials=10, metric='MSE'):
    num_epochs = args.model.num_epochs
    batch_size = args.model.batch_size

    sub_folder = args['date']

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name, sub_folder)):
        os.makedirs(os.path.join(args.main_path, "models", args.model.model_type, args.file_name, sub_folder))

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

    with open(os.path.join(args.main_path,"models", args.model.model_type,
                           args.file_name, sub_folder, 'model_parameters.json'), 'w') as f:
        json.dump(best_parameters, f)

    if metric == 'MSE':
        criterion = nn.MSELoss(reduction='sum')
    elif metric == 'MAE':
        criterion = nn.L1Loss(reduction='sum')
    elif metric == "MaskedLoss":
        criterion = MaskedL1Loss(reduction='sum')

    model, history = train_model(best_parameters, num_epochs, batch_size, X_train, y_train,
                X_test, y_test, criterion, device, n_cpu=n_cpu)

    torch.save(model, f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/model.pt')
    history.to_hdf(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/loss_values.h5',
                   key='loss')


def train_kan(args, X_train, y_train, X_test, y_test, device, n_cpu=1, metric='MSE'):

    num_epochs = args.model.num_epochs
    batch_size = args.model.batch_size

    sub_folder = args['date']

    if not os.path.isdir(os.path.join(args.main_path, "models", args.model.model_type, args.file_name, sub_folder)):
        os.makedirs(os.path.join(args.main_path, "models", args.model.model_type, args.file_name, sub_folder))

    MODEL_PARAMETERS['feature_scaler'] = args.model.feature_scaler
    MODEL_PARAMETERS['target_scaler'] = args.model.target_scaler

    print("Selected hyperparameters: \n", MODEL_PARAMETERS)

    with open(os.path.join(args.main_path,"models", args.model.model_type,
                           args.file_name, sub_folder, 'model_parameters.json'), 'w') as f:
        json.dump(MODEL_PARAMETERS, f)

    if metric == 'MSE':
        criterion = nn.MSELoss(reduction='sum')
    elif metric == 'MAE':
        criterion = nn.L1Loss(reduction='sum')
    elif metric == "MaskedLoss":
        criterion = MaskedL1Loss(reduction='sum')

    model, history = train_model(MODEL_PARAMETERS, num_epochs, batch_size, X_train, y_train,
                X_test, y_test, criterion, device, n_cpu=n_cpu, verbose=True)

    torch.save(model, f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/model.pt')
    history.to_hdf(f'{args.main_path}/models/{args.model.model_type}/{args.file_name}/{sub_folder}/loss_values.h5',
                   key='loss')