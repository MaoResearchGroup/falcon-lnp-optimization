import numpy as np
import pandas as pd
import pickle
import time
from falcon_engine.utilities import print_slowly


from .search_algorithms import OptimizationSearch
import shap

def run_optimization_pipeline(opt_methods, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME, diversity_threshold, raw_bounds):
    """
    Run optimization for the given cell types and method.
    """
    not_initiliazed = True
    optimizer = None
    optimized_formulations = pd.DataFrame()

    for opt_method in opt_methods:

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
        scaled_bounds = {}  # final scaled bounds per cell_type
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


            # Scale bounds for search algorithms
            scaler = input_scalars[cell_type]
            cell_scaled_bounds = []
            

            for param in input_param_names:
                raw_min, raw_max = raw_bounds[param]
                scaled_min = scaler[param].transform([[raw_min]])[0][0]
                scaled_max = scaler[param].transform([[raw_max]])[0][0]
                cell_scaled_bounds.append((scaled_min, scaled_max))

            scaled_bounds[cell_type] = cell_scaled_bounds

        print("\n--- Raw (Unscaled) Bounds for Each Input Parameter ---")
        for param in input_param_names:
            if param in raw_bounds:
                low, high = raw_bounds[param]
                print(f"{param}: ({low:.4f}, {high:.4f})")
            else:
                print(f"{param}: [BOUND NOT FOUND]")
        
        #max feature importance calculations for cell types 
        shap_ct1 = shap_analysis(RUN_NAME,cell_type_list[0],pipeline)
        shap_ct2 = shap_analysis(RUN_NAME,cell_type_list[1],pipeline)
        max_feature_importance = np.maximum(shap_ct1, shap_ct2)

        #initialize search class
        if (not_initiliazed): 
            not_initiliazed = False
            optimizer = OptimizationSearch(models, 
                                        input_scalars, 
                                        output_scalars, 
                                        training_data, 
                                        input_param_names,
                                            cell_type_list, 
                                            max_feature_importance, 
                                            diversity_threshold, 
                                            opt_method, 
                                            scaled_bounds[cell_type])
            

        # Run optimization for each method
        if opt_method == "DA":
            optimizer.opt_method = opt_method
            suggested_LNPs = optimizer.run_dual_annealing(num_formulations)
        elif opt_method =="BO":
            optimizer.opt_method = opt_method
            suggested_LNPs = optimizer.run_bayesian(num_formulations)
        elif opt_method =="i-optimal":
            optimizer.opt_method = opt_method
            suggested_LNPs = optimizer.run_i_optimal(num_formulations)
        elif opt_method =="NSGAII":
            optimizer.opt_method = opt_method
            suggested_LNPs = optimizer.run_nsga2(num_formulations, 
                                                MAX_cell_targets, 
                                                MIN_cell_targets)
        elif opt_method == "i-optimal_minCT":
            optimizer.opt_method = opt_method
            suggested_LNPs = optimizer.run_i_optimal_minCT(num_formulations)
        else:
            raise KeyError
        
        #Save as .pkl and as excel for user
        suggested_LNPs.to_csv(f'output/{RUN_NAME}/{opt_method}_valid_suggestions.csv', index=False)
        optimized_formulations = pd.concat([optimized_formulations, suggested_LNPs], ignore_index=True)
        print(optimized_formulations)

    return optimized_formulations

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
