import sys, os
import torch

sys.path.append(os.path.abspath("./src"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import hydra
from omegaconf import OmegaConf

from src.models import model_optimisation, model_train
from src.evaluation import return_scores
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
        pass
    else:
        print("Optimisation Results and Evaluation not required!")

    if args.validation_results:
        pass
    else:
        print("Validation Step with Simulation of Optimal Solution Candidates not required!")
        print("--" * 40)


if __name__ == "__main__":
    main()
