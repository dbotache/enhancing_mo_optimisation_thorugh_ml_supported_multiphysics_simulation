from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd


def pareto_performance(y, loc_target_x, loc_target_y, front_list,
                       reference=None, metrics_names=None, labels=None):
    metrics_all_names = ['GD', 'GD+', 'IGD', 'IGD+', 'HV']
    metrics = list([GD, GDPlus, IGD, IGDPlus, HV])

    metrics_dict = dict()

    for i, metric in enumerate(metrics_all_names):
        metrics_dict[metric] = metrics[i]

    if metrics_names is None:
        metrics_names = ['GD', 'GD+', 'IGD', 'IGD+', 'HV']
    if reference is None:
        reference = [1, 1]

    scaler = MinMaxScaler()
    scaler.fit(y.loc[:, [loc_target_x, loc_target_y]])

    metric_values = []
    for j in metrics_names:
        if j != 'HV':
            ind = metrics_dict[j](scaler.transform(y.loc[:, [loc_target_x, loc_target_y]]))
        else:
            ind = metrics_dict[j](reference)

        loc_metrics = []
        for k, loc_front in enumerate(front_list):
            metric_value = ind(scaler.transform(loc_front.loc[:, [loc_target_x, loc_target_y]]))
            loc_metrics.append(metric_value)

        loc_metrics.append(ind(scaler.transform(y.loc[:, [loc_target_x, loc_target_y]])))

        metric_values.append(np.array(loc_metrics))

    if labels is not None:
        return pd.DataFrame(np.vstack(metric_values).T, columns=metrics_names, index=labels)
    else:
        return pd.DataFrame(np.vstack(metric_values).T, columns=metrics_names)








