import sys

sys.path.append("../../src")

import json
import os
import pickle
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('../../src'))

from data_utils.scaling import FeatureTargetScaling


class ModelLoader:
    def __init__(self, main_path, file_name, model_type, model_sub_folder=None,
                 device=torch.device("cpu"), n_cpu=1, verbose=False):
        """
        Module that loads a trained machine learning model, and makes predictions on test data.
        The script requires the following:

        - main_path: The path to the main directory.
        - file_name: The name of the file containing the test data.
        - model_type: The type of machine learning model. Supported models are 'xgb', 'ensemble', 'mlp', and 'cnn'.
        - model_sub_folder (optional): The name of the sub-folder where the trained model is stored. Default is None.
        - device (optional): The device to use for running the model. Default is CPU.
        - n_cpu (optional): The number of CPU cores to use. Default is 1.

        The script loads the trained model and any associated model parameters, and then applies it
        to the test data to make predictions.

        Args:
        main_path (str): Path to the data directory.
        file_name (str): Name of the file used in training process.
        model_type (str): Type of the model ('xgb', 'ensemble', 'mlp', or 'cnn').
        model_sub_folder (str, optional): Sub-folder name for the model. Default is None.
        device (torch.device, optional): The device to run the model on. Default is 'cpu'.
        n_cpu (int, optional): Number of CPUs to use. Default is 1.

        Attributes:
        target_names (list): Names of the target variables.
        target_names_of_models_available (list): Names of the available target variables for the model.
        train_features (pandas.DataFrame): Training features.
        train_targets (pandas.DataFrame): Training targets.
        test_features (pandas.DataFrame): Test features.
        test_targets (pandas.DataFrame): Test targets.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        scaler_object (data_utils.scaling.FeatureTargetScaling): The scaler object for scaling the features and targets.
        single_model (torch.nn.Module): The trained model for a single target.
        model_container (list): List of trained models for all available targets.
        model_parameter_container (list): List of parameters for all trained models.


        Functions:

        - get_target_names(): Gets the names of the target variables from a JSON file containing meta-information.
        - load_xgb_ensemble(): Loads an XGBoost or ensemble model for each target variable.
        - load_mlp_cnn(): Loads a PyTorch MLP or CNN model.
        - read_test_data(): Reads the test data from disk.
        - get_scaler(): Scales the test features using the same scaling factor as used during training.
        - predict(): Uses the trained model to make predictions on the test data.

        """
        self.main_path = main_path
        self.file_name = file_name
        self.model_type = model_type
        self.model_sub_folder = model_sub_folder
        self.device = device
        self.n_cpu = n_cpu
        self.verbose = verbose

        self.target_names = []
        self.target_names_of_models_available = []

        self.single_model = None
        self.model_container = []
        self.model_parameter_container = []

        self.get_target_names()
        self.read_model()

    def read_model(self):
        if self.model_type == "xgb" or self.model_type == "ensemble":
            self.load_xgb_ensemble()
        elif self.model_type == "mlp" or self.model_type == "cnn":
            self.load_mlp_cnn()
        else:
            print("model not implemented")

    def load_xgb_ensemble(self):
        model_path = None
        parameter_path = None
        if self.model_sub_folder is None:
            model_dir = f"{self.main_path}/models/{self.model_type}/{self.file_name}"
        else:
            model_dir = f"{self.main_path}/models/{self.model_type}/{self.model_sub_folder}/{self.file_name}"

        for target_name in self.target_names:
            if self.model_type == "xgb":
                model_path = f"{model_dir}/{target_name}.pkl"
                parameter_path = f"{model_dir}/{target_name}_parameters.json"
            elif self.model_type == "ensemble":
                model_path = f"{model_dir}/{target_name}/ensemble.pkl"

            if os.path.isfile(model_path):
                with open(model_path, "rb") as f:
                    reg_ = pickle.load(f)
                self.model_container.append(reg_)
                self.target_names_of_models_available.append(target_name)
            else:
                print(f"{self.model_type} model not available for target: {target_name}")
                print(model_path)

            if isinstance(parameter_path, str) and os.path.isfile(parameter_path):
                with open(parameter_path, "r") as fp:
                    parameter = json.load(fp)
                self.model_parameter_container.append(parameter)
            else:
                if self.verbose:
                    print("No model parameter json file for ", self.model_type, ' - ', target_name)

    def load_mlp_cnn(self):
        if self.model_sub_folder is None:
            model_dir = f"{self.main_path}/models/{self.model_type}/{self.file_name}"
        else:
            model_dir = f"{self.main_path}/models/{self.model_type}/{self.model_sub_folder}/{self.file_name}"

        model = torch.load(f"{model_dir}/model.pt", map_location=self.device)
        parameter_path = f"{model_dir}/model_parameters.json"

        if os.path.isfile(parameter_path):
            with open(parameter_path, "r") as fp:
                parameter = json.load(fp)
            self.model_parameter_container.append(parameter)
            if self.verbose:
                print(f"Feature scaler for Model: {self.model_type}: {self.model_parameter_container[0]['feature_scaler']}")
                print(f"Target scaler for Model: {self.model_type}: {self.model_parameter_container[0]['target_scaler']}")

        else:
            if self.verbose:
                print(f"No model parameter json file for {model_type} in ", model_dir)

        self.single_model = model.to(self.device)

    def get_target_names(self):
        json_file = os.path.join(self.main_path, f"{self.file_name}.json")
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                config = json.load(f)
            self.target_names = config["feature_names"]["targets"]
        else:
            print("there is no meta information of file: ", self.file_name)
            print("please add meta [.json file] with features_names and target_names!")

    def predict(self, test_data, scaler_object=None):
        if self.model_type == "xgb" or self.model_type == "ensemble":
            return self.predict_sklearn(df=test_data, scaler_object=scaler_object)
        elif self.model_type == "mlp" or self.model_type == "cnn":
            return self.predict_pytorch(df=test_data, scaler_object=scaler_object)

    def predict_sklearn(self, df, scaler_object=None):

        y_pred_list = []
        for i, target_name in enumerate(self.target_names_of_models_available):
            if scaler_object:
                df = scaler_object[i].scale_data(df)
            if self.model_type == "ensemble":
                y_pred_list.append(self.model_container[i].predict(df))
            elif self.model_type == "xgb":
                outputs = self.model_container[i].predict(df.values)
                y_pred_list.append(outputs)

        pred_array = np.vstack(y_pred_list).T
        pred_df = pd.DataFrame(pred_array, columns=self.target_names_of_models_available, index=df.index)

        if scaler_object:
            pred_df = scaler_object[i].inverse_transform_targets(pred_df)

        return pred_df

    def predict_pytorch(self, df, scaler_object=None):

        if isinstance(df, pd.DataFrame):
            if scaler_object:
                df = scaler_object[0].scale_data(df)

            tensor = torch.tensor(df.values, dtype=torch.float32)
            if self.model_type == 'cnn':
                tensor = tensor.reshape(-1, 1, df.shape[1])
            predictions_tensor = self.predict_pytorch_tensor(tensor).cpu()
            predictions = pd.DataFrame(predictions_tensor, columns=self.target_names, index=df.index)

            if scaler_object:
                predictions = scaler_object[0].inverse_transform_targets(predictions)

        else:
            raise ValueError("Data must be a pandas DataFrame.")

        return predictions

    def predict_pytorch_tensor(self, data):
        with torch.no_grad():
            output = self.single_model(data.to(self.device))
        return output

    def predict_pytorch_dataloader(self, dataloader):
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                output = self.single_model(batch.to(self.device))
                predictions.extend(output.tolist())
        return predictions


class ModelWrapper:
    """
    A class to load and manage multiple machine learning models.

    Args:
    main_path (str): The path to the directory containing the model files.
    file_name (str): The name of the file(s) related to the model to be loaded.
    model_types (list): A list of strings representing the types of models to be loaded.
    model_sub_folder (str, optional): The sub-folder within the main_path containing the model files.
    device (torch.device, optional): The device to be used for model inference. Defaults to torch.device('cpu').
    n_cpu (int, optional): The number of CPUs to be used for model inference. Defaults to 1.

    Attributes:
    device (torch.device): The device to be used for model inference.
    model_objects (list): A list of model objects, one for each model type specified in model_types.
    """

    def __init__(self, main_path, file_name, model_types, model_sub_folder=None,
                 device=torch.device("cpu"), n_cpu=1, verbose=False):
        """
        Initializes a ModelWrapper object.

        Args:
        main_path (str): The path to the directory containing the model files.
        file_name (str): The name of the file(s) used in training process to be loaded.
        model_types (list): A list of strings representing the types of models to be loaded.
        model_sub_folder (str, optional): The sub-folder within the main_path containing the model files.
        device (torch.device, optional): The device to be used for model inference. Defaults to torch.device('cpu').
        n_cpu (int, optional): The number of CPUs to be used for model inference. Defaults to 1.
        """
        self.main_path = main_path
        self.file_name = file_name
        self.device = device
        self.model_objects = []
        self.scaler_objects = []
        self.train_features = pd.DataFrame()
        self.train_targets = pd.DataFrame()

        for m_type in model_types:
            self.model_objects.append(
                ModelLoader(
                    main_path=self.main_path,
                    file_name=self.file_name,
                    model_type=m_type,
                    model_sub_folder=model_sub_folder,
                    device=self.device,
                    n_cpu=n_cpu,
                    verbose=verbose
                )
            )

        self.get_scaler_objects()

    def read_data(self, sub_folder="train_80perc_test_20perc"):
        file_path = os.path.join(self.main_path, "Dataframes", sub_folder)
        if not os.path.isdir(file_path):
            print("Not a directory:", file_path)

        self.train_features = pd.read_hdf(f"{file_path}/X_train_{self.file_name}.h5")
        self.train_targets = pd.read_hdf(f"{file_path}/y_train_{self.file_name}.h5")

    def get_scaler_objects(self, X=None, y=None):

        if X is None and y is None:
            self.read_data()
            X = self.train_features
            y = self.train_targets

        for model_object in self.model_objects:
            scaler_list = []
            for parameters in model_object.model_parameter_container:
                x_y_scaling = FeatureTargetScaling(X, y,
                                                   scaler_type_features=parameters['feature_scaler'],
                                                   scaler_type_targets=parameters['target_scaler'])
                scaler_list.append(x_y_scaling)
            self.scaler_objects.append(scaler_list)

        return self.scaler_objects

    def return_predictions(self, df):
        """
        Returns a list of predictions for the given test data, one prediction per model.

        Args:
        df (DataFrame): The data to be used for making predictions

        Returns:
        list: A list of predictions, one per model.
        """

        pred_list = []
        for i, model_object in enumerate(self.model_objects):
            if self.scaler_objects[i]:
                pred_list.append(model_object.predict(test_data=df, scaler_object=self.scaler_objects[i]))
            else:
                pred_list.append(model_object.predict(test_data=df))
        return pred_list


if __name__ == "__main__":
    main_path = "/home/dbotache/datasets/kite/parameter_designs"
    file_name = "DoE_29V_791D"
    model_type = "mlp"
    model_sub_folder = None  # '2023_03_15_Archiv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use ModelLoader Class
    model_object = ModelLoader(
        main_path=main_path,
        file_name=file_name,
        model_type=model_type,
        model_sub_folder=model_sub_folder,
        device=device,
    )

    pred = model_object.predict()
    print(pred.shape)

    # use ModelWrapper Class
    wrapper = ModelWrapper(
        main_path=main_path,
        file_name=file_name,
        model_types=["ensemble", "xgb", "mlp"],
        model_sub_folder=model_sub_folder,
    )

    print(len(wrapper.return_predictions()))
