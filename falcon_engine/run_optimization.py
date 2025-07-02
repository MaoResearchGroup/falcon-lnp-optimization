import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from bayes_opt import BayesianOptimization

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

import pickle
import time
from falcon_engine.utilities import print_slowly



from .search_algorithms import OptimizationSearch
import shap

def run_optimization_pipeline(opt_method, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME, diversity_threshold, norm_suggestion_bounds = (-0.1, 1.2)):
    """
    Run optimization for the given cell types and method.
    """

    print_slowly('\n######### DE NOVO FORMULATION SEARCH #####')
    print_slowly(f"Optimization Algorithm: {opt_method}")
    print_slowly(f"Number of Formulations: {num_formulations}")


    if opt_method in ['DA', 'BO']:
        print_slowly(f"Max Cell Target: {MAX_cell_targets[0]}")
    else:
        print_slowly(f"Max Cell Targets: {MAX_cell_targets}")
        print_slowly(f"Min Cell Targets: {MIN_cell_targets}")

    start_time = time.time()

    # load the model and scalers 
    # store models in dictionary 
    cell_type_list = MAX_cell_targets + MIN_cell_targets # we want to make single objective predictions for MAX_cell_targfets
    models = {}
    input_scalars = {}
    output_scalars = {}
    training_data = {}
    pipeline = None

    for cell_type in cell_type_list:
        model_path = f'output/{RUN_NAME}/{cell_type}/'
        with open(f'{model_path}Pipeline_dict.pkl', 'rb') as file:
            pipeline = pickle.load(file)

        models[cell_type] = pipeline['Model_Selection']['Best_Model']['Model']
        output_scalars[cell_type] = pipeline['Data_preprocessing']['Output_Scaler']
        input_scalars[cell_type] = pipeline['Data_preprocessing']['Scalers']
        training_data[cell_type] = pipeline['Data_preprocessing']['X']
    input_param_names = pipeline['Data_preprocessing']['Input_Params']

    #shap analysis run for cell type 0 
    feature_importance = shap_analysis(RUN_NAME,cell_type_list[0],pipeline)

    #initialize search class
    optimizer = OptimizationSearch(models, 
                                   input_scalars, 
                                   output_scalars, 
                                   training_data, 
                                   input_param_names,
                                    cell_type_list, 
                                    feature_importance, 
                                    diversity_threshold, 
                                    opt_method, 
                                    norm_suggestion_bounds)

    if opt_method == "DA":
        suggested_LNPs = optimizer.run_dual_annealing(num_formulations)
    elif opt_method =="BO":
        suggested_LNPs = optimizer.run_bayesian(num_formulations)
    elif opt_method =="i-optimal":
        suggested_LNPs = optimizer.run_i_optimal(num_formulations)
    elif opt_method =="NSGAII":
        suggested_LNPs = optimizer.run_nsga2(num_formulations, 
                                             MAX_cell_targets, 
                                             MIN_cell_targets)
    else:
        raise KeyError
    
    #Save as .pkl and as excel for user
    suggested_LNPs.to_csv(f'output/{RUN_NAME}/{opt_method}_valid_suggestions.csv', index=False)

    return suggested_LNPs

# Shap analysis run - outputs 
def shap_analysis(RUN_NAME, cell_type,pipeline):
    # store models in dictionary 
    models = {}
    input_scalars = {}
    output_scalars = {}

    models[cell_type] = pipeline['Model_Selection']['Best_Model']['Model']
    output_scalars[cell_type] = pipeline['Data_preprocessing']['Output_Scaler']
    input_scalars[cell_type] = pipeline['Data_preprocessing']['Scalers']
    train_data = pipeline['Data_preprocessing']['X']
    input_param_names = pipeline['Data_preprocessing']['Input_Params']

    shap_values = {}

    explainer = shap.Explainer(models[cell_type])
    X = pd.DataFrame(train_data, columns=input_param_names)
    shap_values[cell_type] = explainer(X)

    shap_matrix = shap_values[cell_type].values  # shape: (n_samples, n_features)

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

    # Create a pandas Series for easier viewing, matching input_param_names
    feature_importance = pd.Series(mean_abs_shap, index=input_param_names)

    return feature_importance
