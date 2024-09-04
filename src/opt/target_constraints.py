import numpy as np

CONSTRAINTS_TARGETS_BM = [{
    "col_names" : ["Masse_mag"],
    "bounds" : [-np.inf, 3.25]
    }
]

CONSTRAINTS_CFD = [{
    "col_names" : ['pressure_loss'],
    "bounds" : [-np.inf, 1000]
    },
    {
    "col_names" : ['cooling_power'],
    "bounds" : [-np.inf, 1000]
    }
]

