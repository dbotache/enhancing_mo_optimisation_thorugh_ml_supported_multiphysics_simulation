import numpy as np
import pandas as pd
import torch

from pymoo.core.problem import Problem
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

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


def opt_fn_cfd(dict_out):
    return [dict_out["pressure_loss"], dict_out["cooling_power"]]


def opt_fn_motor_free(dict_out):
    # c1 = torch.clamp(3.25 - torch.tensor(dict_out["Masse_mag"]), min=0).numpy()
    return [dict_out["M"] * (-1), dict_out["P_loss_total"], dict_out["Masse_mag"]]


def opt_fn_motor(dict_out, factor_1=1.0, factor_2=1.0):
    # c1 = torch.clamp(323.0 - torch.tensor(dict_out["M(S2,n1)"]), min=0).numpy()
    # c2 = torch.clamp(321.0 - torch.tensor(dict_out["M(S2,n2)"]), min=0).numpy()
    # c3 = torch.clamp(191.0 - torch.tensor(dict_out["M(S2,n3)"]), min=0).numpy()
    # c4 = torch.clamp(94.0 - torch.tensor(dict_out["M(S2,n4)"] ), min=0).numpy()
    o_1 = dict_out["Pv_Antrieb_Fzg_Zykl_1"] * factor_1
    o_2 = dict_out["Pv_Antrieb_Fzg_Zykl_2"] * factor_1
    o_3 = dict_out["MEK_Aktivteile"] * factor_2

    c_naiv = dict_out["M(S2,n1)"] * -1
    c1 = torch.clamp(330.0 - torch.tensor(dict_out["M(S2,n1)"]), min=0).numpy()
    c2 = torch.clamp(330.0 - torch.tensor(dict_out["M(S2,n2)"]), min=0).numpy()
    c3 = torch.clamp(200.0 - torch.tensor(dict_out["M(S2,n3)"]), min=0).numpy()
    c4 = torch.clamp(100.0 - torch.tensor(dict_out["M(S2,n4)"]), min=0).numpy()
    c = c1 + c2 + c3 + c4

    return [c, (o_1 + o_2) / 2, o_3]


""" deprecated: 2024 04 26
def opt_fn_motor(dict_out, factor_1=1.0, factor_2=1.0):
    c1 = torch.clamp(323.0 - torch.tensor(dict_out["M(S2,n1)"]), min=0).numpy()
    c2 = torch.clamp(321.0 - torch.tensor(dict_out["M(S2,n2)"]), min=0).numpy()
    c3 = torch.clamp(191.0 - torch.tensor(dict_out["M(S2,n3)"]), min=0).numpy()
    c4 = torch.clamp(94.0 - torch.tensor(dict_out["M(S2,n4)"] ), min=0).numpy()

    o_1 = dict_out["Pv_Antrieb_Fzg_Zykl_1"] * factor_1
    o_2 = dict_out["Pv_Antrieb_Fzg_Zykl_2"] * factor_1
    o_3 = dict_out["MEK_Aktivteile"] * factor_2
    obj = c1 + c2 + c3 + c4 + o_1 + o_2 + o_3

    return [obj, dict_out["Pv_Antrieb_Fzg_Zykl_1"], dict_out["MEK_Aktivteile"]]
    """


class MyOutput_motor(Output):
    def __init__(self):
        super().__init__()
        self.obj1min = Column("obj_min", width=13)
        self.obj1mean = Column("obj_mean", width=13)
        self.obj1var = Column("obj_var", width=13)
        self.obj2min = Column("Pv_A_Z_1_min", width=13)
        self.obj2mean = Column("Pv_A_Z_1_mean", width=13)
        self.obj2var = Column("Pv_A_Z_1_var", width=13)
        self.obj3min = Column("MEK_Akt_min", width=13)
        self.obj3mean = Column("MEK_Akt_mean", width=13)
        self.obj3var = Column("MEK_Akt_var", width=13)
        self.columns += [self.obj1min, self.obj1mean, self.obj1var, self.obj2min, self.obj2mean, self.obj2var,
                         self.obj3min, self.obj3mean, self.obj3var]

    def update(self, algorithm):
        super().update(algorithm)
        self.obj1min.set(algorithm.pop.get("F")[:, 0].min())
        self.obj1mean.set(algorithm.pop.get("F")[:, 0].mean())
        self.obj1var.set(algorithm.pop.get("F")[:, 0].var())
        self.obj2min.set(algorithm.pop.get("F")[:, 1].min())
        self.obj2mean.set(algorithm.pop.get("F")[:, 1].mean())
        self.obj2var.set(algorithm.pop.get("F")[:, 1].var())
        self.obj3min.set(algorithm.pop.get("F")[:, 2].min())
        self.obj3mean.set(algorithm.pop.get("F")[:, 2].mean())
        self.obj3var.set(algorithm.pop.get("F")[:, 2].var())


class MyOutput_cfd(Output):
    def __init__(self):
        super().__init__()
        self.obj1min = Column("obj_p_min", width=13)
        self.obj1mean = Column("obj_p_mean", width=13)
        self.obj1var = Column("obj_p_var", width=13)
        self.obj2min = Column("obj_T_min", width=13)
        self.obj2mean = Column("obj_T_mean", width=13)
        self.obj2var = Column("obj_T_var", width=13)
        self.columns += [self.obj1min, self.obj1mean, self.obj1var, self.obj2min, self.obj2mean, self.obj2var]

    def update(self, algorithm):
        super().update(algorithm)
        self.obj1min.set(algorithm.pop.get("F")[:, 0].min())
        self.obj1mean.set(algorithm.pop.get("F")[:, 0].mean())
        self.obj1var.set(algorithm.pop.get("F")[:, 0].var())
        self.obj2min.set(algorithm.pop.get("F")[:, 1].min())
        self.obj2mean.set(algorithm.pop.get("F")[:, 1].mean())
        self.obj2var.set(algorithm.pop.get("F")[:, 1].var())
