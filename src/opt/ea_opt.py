#%%
import os
import pandas as pd
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from opt.opt_utils import MyProblem


def ea_optimization(args, problem_dict):

    ##Important for future usage: add model_path as variable to config and to load_ensemble function from ea_utils
    ##Write a model wrapper function that MLP and Ensemble models (and other models) work equaly within this pipeline:
    ##adapt Myproblem._evaluate function
    problem = MyProblem(args, problem_dict)
    ##Important: add Hyperparameters to the algorithm
    #define algorithm:
    algorithm = NSGA2(pop_size=args.opt.pop_size,
                      sampling=MixedVariableSampling(),
                      mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                      eliminate_duplicates=MixedVariableDuplicateElimination(),)

    #optimize!!!!:

    res = minimize(problem,
                   algorithm,
                   ('n_gen', args.opt.n_gen),
                   seed=args.random_seed,
                   output=problem_dict["MyOutput"](),
                   save_history=args.opt.save_history,
                   verbose=args.opt.verbose)
    pop = res.pop

    ##Export FeatureValues X for Individuals of the last generation
    X_lastpop = pd.DataFrame(pop.get("X")[i] for i in range(len(pop.get("X"))))
    X_lastpop = X_lastpop.reindex(columns=problem_dict["var_names"].values)

    ##Export the most "promising/important" Individuals of the last generation
    dm = HighTradeoffPoints()
    Idxs = dm(pop.get("F"))
    X_promising = pop.get("X")[Idxs]
    X_promising = pd.DataFrame(X_promising[i] for i in range(len(X_promising)))
    X_promising = X_promising.reindex(columns=problem_dict["var_names"].values)

    #if path does not exist, create it
    if not os.path.exists(args.opt.save_path):
        os.makedirs(args.opt.save_path)
    X_lastpop.to_hdf(f'{args.opt.save_path}/X_{args.file_name}_ea_samples_{args.model.model_type}.h5', key='features')
    X_promising.to_hdf(f'{args.opt.save_path}/X_{args.file_name}_promising_ea_samples_{args.model.model_type}.h5', key='features')

    print(f"Saved two Dataframes to {args.opt.save_path}")



#Maybe for later but not 4 today: JD
#
##%%
#
#plot = Scatter()
#plot.add(pop.get("F"))
#plot.show()
#
#
#dm = HighTradeoffPoints()
#
#I = dm(pop.get("F")[:,:2])
#
#plot = Scatter()
#plot.add(pop.get("F")[:,:2])
#plot.add(pop.get("F")[I,:2], color="red", s=100)
#plot.show()
## %%
#dm = HighTradeoffPoints()
#
#I = dm(pop.get("F")[:,1:3])
#
#plot = Scatter()
#plot.add(pop.get("F")[:,1:3])
#plot.add(pop.get("F")[I,1:3], color="red", s=100)
#plot.show()
#
#
##%%
#len(pop.get("F")[I])
## %%
#
# %%
