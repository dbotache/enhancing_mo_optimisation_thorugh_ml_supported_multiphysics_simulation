from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

def opt_fn_motor_free(dict_out):
    # c1 = torch.clamp(3.25 - torch.tensor(dict_out["Masse_mag"]), min=0).numpy()
    return [dict_out["M"] * (-1), dict_out["P_loss_total"], dict_out["Masse_mag"]]

class MyOutput_motor(Output):
    def __init__(self):
        super().__init__()
        self.obj1min = Column("M_min", width=13)
        self.obj1mean = Column("M_mean", width=13)
        self.obj1var = Column("M_var", width=13)
        self.obj2min = Column("P_loss_min", width=13)
        self.obj2mean = Column("P_loss_mean", width=13)
        self.obj2var = Column("P_loss_var", width=13)
        self.obj3min = Column("Masse_mag_min", width=13)
        self.obj3mean = Column("Masse_mag_mean", width=13)
        self.obj3var = Column("Masse_mag_var", width=13)
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