# GitHub Repository Usage Guide

## Overview

This repository is structured to facilitate the configuration, training, optimization, and evaluation of machine learning models using various YAML configuration files. It is designed with a clear separation of concerns, where each subfolder within the `configs` directory defines the parameters for different components of the machine learning pipeline, including model specifications, optimization algorithms, evaluation methods, and explainability techniques.

This guide will walk you through the repository structure and how to use the YAML configuration files to customize and run different machine learning experiments.

---

## Repository Structure

### 1. `configs` Folder

The `configs` folder contains YAML configuration files, which define all parameters for running and experimenting with machine learning models. This folder is further divided into subfolders that hold the configurations for specific pipeline stages:

- **`eval/`**: Configuration files for evaluation methods.
  - `pareto.yaml`: Configures Pareto-based multi-objective evaluation.
  - `regression.yaml`: Configures regression-based evaluation metrics.

- **`model/`**: Configuration files for various machine learning models.
  - Each file contains the parameters specific to a model type (e.g., `cnn.yaml`, `xgb.yaml`), including hyperparameters, device setup, and training details.

- **`opt/`**: Configuration files for optimization algorithms.
  - `ea_opt.yaml`: Specifies the configuration for evolutionary algorithm (EA) optimization, including number of generations, population size, and verbosity.

- **`xai/`**: Configuration files for explainability techniques.
  - For example, `pdp.yaml` could specify configuration for Partial Dependence Plots (PDP) used to explain the model.

- **`main.yaml`**: The main configuration file, which pulls together all the different components from the subfolders and specifies general pipeline settings, such as the dataset path, the seed for random operations, and which tasks should be executed (e.g., training, regression evaluation, optimization evaluation).

#### `main.yaml` Structure

```yaml
defaults:
  - model: xgb
  - opt: ea_opt
  - eval: pareto
  - xai: pdp
  - _self_

main_path: /home/datasets/parameter_designs
splitfolder: train_80perc_test_20perc
file_name: ${file_name}

random_seed: 42

train_models: False
find_pareto: False
regression_evaluation: False
explain_regression: False
optimisation_evaluation: True
validation_results: False

output_dir: ./output

hydra:
  job:
    chdir: True
  run:
    dir: ${output_dir}
```

Key parameters:
- **defaults**: Specifies the default configurations for model type, optimization method, evaluation approach, and explainability technique. These settings will be inherited from the corresponding YAML files in subfolders.
- **main_path**: Defines the path to the dataset.
- **random_seed**: Ensures reproducibility by setting a random seed.
- **task flags**: Boolean flags to control which parts of the pipeline are executed (e.g., `train_models`, `find_pareto`, `optimisation_evaluation`).
- **output_dir**: Specifies the directory where the output will be saved.
- **hydra**: A framework for managing configuration and experiment directories dynamically.

### 2. `src` Folder

The `src` folder contains the main scripts and implementations of the machine learning pipeline. These scripts are responsible for:
- Loading configurations from the `configs` folder.
- Preprocessing the data.
- Training the models.
- Performing optimization based on the selected algorithm.
- Conducting evaluations and generating explainability reports.

---

## How to Use the Repository

### 1. Configure Your Experiment

To customize an experiment, you can modify the relevant YAML files in the `configs` folder.

- **Model Configuration**: To change the model type or update hyperparameters, edit the corresponding YAML file in the `model` folder. For example, `cnn.yaml` might define the configuration for a Convolutional Neural Network (CNN) model:
  
  ```yaml
  model_type: cnn
  hyperparameter_tuning: True
  feature_scaler: minmax
  target_scaler: minmax
  device: cuda
  opt_trials: 30
  num_epochs: 500
  batch_size: 32
  metric: MAE
  ```

- **Optimization Configuration**: To modify the optimization settings, edit the `ea_opt.yaml` in the `opt` folder. For example:
  
  ```yaml
  opt_type: ea
  n_gen: 100
  pop_size: 2000
  save_history: False
  verbose: True

  save_path: ${file_name}/${splitfolder}/${opt.opt_type}/${model.model_type}/${now:%Y-%m-%d_%H%M%S}
  ```

  This configuration specifies evolutionary algorithm (EA) optimization with 100 generations and a population size of 2000.

- **Evaluation Configuration**: To change the evaluation method, edit the corresponding file in the `eval` folder. For example, `pareto.yaml`:
  
  ```yaml
  model_types: ["xgb", "ensemble", "mlp", "cnn"]
  opt_alg: ea
  ```

  This will evaluate multiple model types based on a Pareto front approach.

### 2. Run the Pipeline

Once the configurations are set, the pipeline can be executed by running the main script in the `src` folder. The pipeline will:
- Read the configuration from the `main.yaml` file.
- Dynamically load model, optimization, evaluation, and XAI configurations from their respective YAML files.
- Execute the pipeline stages based on the boolean flags (e.g., `train_models`, `find_pareto`).

### 3. Output

The results of your experiments will be stored in the directory specified by `output_dir` in `main.yaml`. The file structure inside `output_dir` will reflect the selected optimization and model types as specified in the configurations.

For example, if using evolutionary algorithm optimization and CNN models, the output may be stored in:

```
./output/${file_name}/train_80perc_test_20perc/ea/cnn/2024-09-09_151220/
```

---

## Citation

If you use this pipeline in your research or project, please cite the following paper:

```bibtex
@InProceedings{BDR+2024ECMLPKDD,
  author    = {Botache, Diego and Decke, Jens and Ripken, Winfried and Dornipati, Abhinay and G{\"o}tz-Hahn, Franz and Ayeb, Mohamed and Sick, Bernhard},
  editor    = {Bifet, Albert and Krilavi{\v{c}}ius, Tomas and Miliou, Ioanna and Nowaczyk, Slawomir},
  title     = {Enhancing Multi-objective Optimisation Through Machine Learning-Supported Multiphysics Simulation},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {297--312},
  isbn      = {978-3-031-70381-2}
}
```

Feel free to include this citation in your work when referencing the pipeline provided in this repository.