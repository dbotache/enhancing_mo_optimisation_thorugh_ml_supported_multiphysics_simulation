import sys, os
import torch

sys.path.append(os.path.abspath("./src"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import hydra
from omegaconf import OmegaConf
from src.models import model_optimisation, model_train
from src.evaluation import return_scores, save_pareto_front
from src.evaluation import calculate_pareto_performance
from src.evaluation import validation_step
from src.opt import make_opt
from src.explainability import explain_model


@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(args):
    print("--" * 40)

    print(OmegaConf.to_yaml(args))

    print("--" * 40)

    if args.train_models:
        if args.model.hyperparameter_tuning:
            model_optimisation(args)
            print(f"Model optimization using {args.model.model_type} finished")
        else:
            model_train(args)
            print(f"Model training using {args.model.model_type} finished")
    else:
        print("Models are not trained, they are already there!")
        print("--" * 40)

    if args.regression_evaluation:
        scores_df = return_scores(args)
        print("Regression Evaluation:")
        print(scores_df)
    else:
        print("Regression Evaluation not required!")

    if args.explain_regression:
        print("xAI-Evaluation:")
        explain_model(args)
    else:
        print("Regression xAI-Evaluation not required!")

    if args.find_pareto:
        print("Optimize and looking for better design parameter sets")
        make_opt(args)
    else:
        print("dont need to look for better design parameters")
        print("--" * 40)

    if args.optimisation_evaluation:
        print("Saving Pareto-Frontiers")
        front_list = save_pareto_front(args)
        print("Pareto performance based on model predictions: ")
        print("Given-Index for available models and last index correspond to Database")
        calculate_pareto_performance(args, front_list)
    else:
        print("Optimisation evaluation not required!")

    if args.validation_results:
        print("Validation scores: ")
        validation_step(args)
    else:
        print("Validation Step with Simulation of Optimal Solution Candidates not required!")
        print("--" * 40)


if __name__ == "__main__":
    main()
