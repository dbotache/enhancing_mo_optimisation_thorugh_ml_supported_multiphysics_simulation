import numpy as np
import pandas as pd
import torch

from pymoo.core.problem import Problem
from models.model_utils import ModelLoader


class MyProblem(Problem):
    def __init__(self, args, problem_dict, **kwargs):
        self.main_path = args.main_path
        self.file_name = args.file_name
        self.model_type = args.model.model_type
        self.target_str_list = problem_dict["target_str_list"]
        self.obj_fn = problem_dict["opt_fn"]
        self.var_names = problem_dict["var_names"]
        variables = problem_dict["variables"]
        n_obj = problem_dict["n_obj"]

        super().__init__(vars=variables, n_obj=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = pd.DataFrame(x[k] for k in range(len(x)))
        x = x.reindex(columns=self.var_names.values)
        model = ModelLoader(self.main_path, self.file_name, self.model_type)
        obj = model.predict(x)
        out["F"] = self.obj_fn(obj)
