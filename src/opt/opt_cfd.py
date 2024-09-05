from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

def opt_fn_cfd(dict_out):
    return [dict_out["pressure_loss"], dict_out["cooling_power"]]

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