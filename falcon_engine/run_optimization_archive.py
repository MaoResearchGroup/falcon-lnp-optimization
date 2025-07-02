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

#shap analysis
import shap

def run_optimization_pipeline(opt_method, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME, diversity_threshold, norm_suggestion_bounds = (-0.1, 1.2)):
    """
    Run optimization for the given cell types and method.
    """
    all_evaluations = [] # List to store all evaluations
    optimized_formulations = [] # List to store optimized formulations and corresponding RLU

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
    

    #Suggested LNP collection
    selected = []
    selected_xy = []
    selected_reversed = []
    test_selected = pd.DataFrame(columns = input_param_names)

    if opt_method == 'DA': 
        print_slowly(f"\n\n--- STARTING DUAL ANNEALING OPTIMIZATION for high {cell_type_list [0]} transfection ---")
        #set search bounds for optimization 
        bounds = [norm_suggestion_bounds for _ in range(len(input_param_names))] # expanded parameter bounds

        #Search loop
        while(len(optimized_formulations) < num_formulations):
            history = [] # save top annealing searches
            history.clear()
            
            result = dual_annealing(
                func=objective_fcn_DA,
                bounds=bounds,
                args=(models[cell_type_list[0]], input_param_names, output_scalars[cell_type_list[0]], all_evaluations),
                initial_temp=20000, 
                visit = 2.8
            )
            maximal_x, maximal_y = result.x, -result.fun

            #convert parameters to physical values
            reversed_x = [input_scalars[cell_type_list[0]][input_param_names[i]].inverse_transform([[maximal_x[i]]])[0][0] for i in range(len(maximal_x))]
            reversed_y = output_scalars[cell_type_list[0]].inverse_transform(np.array(maximal_y).reshape(-1, 1))[0][0]




            #Check if formulations are physically possible
            if (valid_formulation(reversed_x, input_param_names)): #Check if formulations are physically possible

                #Check if formulations are diverse compare to historical and other selected
                if (shap_euc_exclusion(maximal_x, feature_importance, diversity_threshold, training_data[cell_type_list[0]], test_selected)):
                    test_selected.loc[len(test_selected)] = maximal_x
                    selected.append(maximal_x)
                    selected_xy.append(np.append(maximal_x, reversed_y).astype(float) )
                    selected_reversed.append(reversed_x)
                    optimized_formulations.append((reversed_x, reversed_y, opt_method))
                    #print the optimized formulation
                    formatted_params = ", ".join(
                        f"{name}: {value:.3f}" for name, value in zip(input_param_names, reversed_x)
                    )
                    print_slowly(f"Optimized Formulation {len(optimized_formulations)}: {formatted_params}, Predicted LnRLU: {reversed_y}") 
        
        
        #FOR EXPORT
        labeled_reversed = pd.DataFrame(selected_reversed, columns = input_param_names)
        labeled_selected = pd.DataFrame(selected_xy, columns= input_param_names + [f'Predicted_LnRLU'])
        all_evaluations = labeled_selected


    elif opt_method == 'BO':
        print_slowly(f"\n\n--- STARTING BAYESIAN OPTIMIZATION for high {cell_type_list [0]} transfection ---")
        pbounds = {f'param{i}': norm_suggestion_bounds for i in range(1, len(input_param_names)+1)} #expanded parameter bounds 
        while(len(optimized_formulations) < num_formulations): 
            optimizer = BayesianOptimization(
                f=lambda **params: objective_fcn_BO(
                    model=models[cell_type_list[0]],
                    input_param_names=input_param_names,
                    output_scaler = output_scalars[cell_type_list[0]],
                    all_evaluations=all_evaluations,
                    **params
                ),
                pbounds=pbounds,
                verbose=2 
            )
            optimizer.maximize(
                init_points=10,  
                n_iter=50
            )

            #Get best suggested LNPs
            maximal_x, maximal_y = np.array(list(optimizer.max['params'].values())), [optimizer.max['target']] 
            
            #convert parameters to physical values
            reversed_x = [input_scalars[cell_type_list[0]][input_param_names[i]].inverse_transform([[maximal_x[i]]])[0][0] for i in range(len(maximal_x))]
            reversed_y = output_scalars[cell_type_list[0]].inverse_transform(np.array(maximal_y).reshape(-1, 1))[0][0]

            #Check if formulations are physically possible
            if (valid_formulation(reversed_x, input_param_names)): 
                if (shap_euc_exclusion(maximal_x, feature_importance, diversity_threshold, training_data[cell_type_list[0]], test_selected)):
                    test_selected.loc[len(test_selected)] = maximal_x
                    selected.append(maximal_x)
                    selected_xy.append(np.append(maximal_x, reversed_y).astype(float) )
                    selected_reversed.append(reversed_x)


                    optimized_formulations.append((reversed_x, reversed_y, opt_method))
                    #print the optimized formulation
                    formatted_params = ", ".join(
                        f"{name}: {value:.3f}" for name, value in zip(input_param_names, reversed_x)
                    )
                    print_slowly(f"Optimized Formulation {len(optimized_formulations)}: {formatted_params}, Predicted LnRLU: {reversed_y}") 
        
        #FOR EXPORT
        labeled_reversed = pd.DataFrame(selected_reversed, columns = input_param_names)
        labeled_selected = pd.DataFrame(selected_xy, columns= input_param_names + [f'Predicted_LnRLU'])
        all_evaluations = labeled_selected
        

    elif opt_method == 'NSGAII':
        print_slowly(f"\n\n--- STARTING NSGA-II OPTIMIZATION for high {MAX_cell_targets} and low {MIN_cell_targets} transfection ---")
        
        sampling = LHS()  # Latin Hypercube Sampling
        crossover = AdaptiveCrossover(base_prob=0.9, eta=15)
        mutation = AdaptiveMutation(base_prob=0.1, eta=20)
        direction = [-1] * len(MAX_cell_targets) + [1] * len(MIN_cell_targets)
        ordered_models = [models[cell] for cell in cell_type_list]
        
        optimized_formulations = pd.DataFrame()
        max_seeds = 20
        seeds_tested = 0
        all_evaluations = []

        while len(optimized_formulations) < num_formulations and seeds_tested < max_seeds:
            seed = seeds_tested + 1
            seeds_tested += 1

            problem = FormulationOptimizationProblem(
                direction, input_param_names, *ordered_models, pbounds=norm_suggestion_bounds
            )
            algorithm = NSGA2(
                pop_size=500,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True
            )

            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', 20), #used to be 250
                seed=seed,
                save_history=True,
                verbose=True
            )

            print_slowly(f"NSGA-II optimization with seed {seed} completed, exporting result...")

            all_F = np.vstack([np.array([ind.F for ind in gen.pop]) for gen in res.history])
            export_dict = {
                'pareto_F': res.F,
                'all_F': all_F,
                'cell_type_names': cell_type_list,
                'pareto_X': res.X,
                'direction': direction
            }
            with open(f'output/{RUN_NAME}/NSGAII_results_seed{seed}.pkl', "wb") as f:
                pickle.dump(export_dict, f)

            all_X = np.vstack([gen.pop.get("X") for gen in res.history])
            for cell_type in cell_type_list:
                model = models[cell_type]
                output_scaler = output_scalars[cell_type]
                y_preds = model.predict(all_X)
                y_reversed = output_scaler.inverse_transform(y_preds.reshape(-1, 1)).flatten()

                for i, x in enumerate(all_X):
                    if i >= len(all_evaluations):
                        all_evaluations.append({input_param_names[j]: x[j] for j in range(len(x))})
                    all_evaluations[i][f'Predicted_LnRLU_{cell_type}'] = y_reversed[i]

            # Extract Pareto front
            print_slowly("Extracting Pareto optimal solutions...")
            pop = res.pop.get("X")
            obj = res.pop.get("F")
            front = NonDominatedSorting().do(obj, only_non_dominated_front=True)
            pareto_solutions = pop[front]

            # Greedy selection on Pareto solutions
            newly_selected = greedy_selection(
                pareto_solutions,
                num_formulations - len(optimized_formulations),
                input_param_names,
                models,
                input_scalars,
                output_scalars,
                cell_type_list,
                opt_method,
                feature_importance,
                training_data[cell_type_list[0]],
                diversity_threshold
            )

            # Append new selections
            optimized_formulations = pd.concat([optimized_formulations, newly_selected], ignore_index=True)

        if len(optimized_formulations) < num_formulations:
            print(f"\nWARNING: Only {len(optimized_formulations)} formulations found after {max_seeds} seeds.")
        else:
            print(f"\nSUCCESS: {len(optimized_formulations)} optimized formulations selected.")
        all_evaluations = optimized_formulations

    elif opt_method == 'i-optimal':
        print_slowly(f"\n\n--- STARTING i-optimal OPTIMIZATION for IMPROVED MODEL PREDICTIONS ---")

        # Load model and training data
        pred_model = models[cell_type_list[0]]
        X_train = training_data[cell_type_list[0]]

        # Fit surrogate GP
        gp_on_xgb = fit_gp_on_xgb_output(X_train, pred_model)

        # Define bounds
        pbounds = [(-0.1, 1.2)] * len(input_param_names)

        # Latin hypercube sampling
        sampler = qmc.LatinHypercube(len(pbounds))
        X_candidates = qmc.scale(sampler.random(1000), [b[0] for b in pbounds], [b[1] for b in pbounds])
        X_eval = qmc.scale(sampler.random(500), [b[0] for b in pbounds], [b[1] for b in pbounds])

        # Initialize selector
        selector = DiverseValidSelector(
            input_param_names=input_param_names,
            models=models,
            input_scalars=input_scalars,
            output_scalars=output_scalars,
            cell_type_list=cell_type_list,
            feature_importance=feature_importance,
            training_data=training_data[cell_type_list[0]],
            diversity_threshold=diversity_threshold,
            opt_method=opt_method
        )

        count = 0
        while len(selector.optimized_formulations) < num_formulations and len(X_candidates) > 0:
            count += 1
            if count >= 100:
                print(f"Max (100) searches reached; {len(selector.optimized_formulations)} valid formulations found.")
                break

            # Search for best candidate based on i-optimality
            suggested_x, min_avg_var = i_optimal_search(X_train, X_candidates, X_eval, pred_model, gp_on_xgb, selector.selected_normalized)

            # Remove selected candidate from pool
            X_candidates = np.delete(X_candidates, np.where((X_candidates == suggested_x).all(axis=1))[0], axis=0)

            # Try adding point to the selector
            added = selector.try_add_point(suggested_x, verbose=True)
            if added:
                print_slowly(f"i-optimal selected LNP {len(selector.optimized_formulations)} variance: {min_avg_var}")

        # Get final output
        all_evaluations = selector.get_optimized_formulations()
      
    print_slowly("\n\n--- %s minutes for OPTIMIZED FORMULATION GENERATION ---" % ((time.time() - start_time)/60))

    print_slowly("\n\n--- EXPORTING ALL EVALUATIONS to csv ---")
    # Save all evaluations to a CSV file
    
    evaluations_df = pd.DataFrame(all_evaluations)
    evaluations_df.to_csv(f'output/{RUN_NAME}/{opt_method}_all_evaluations.csv', index=False)
    labeled_reversed.to_csv(f'output/{RUN_NAME}/{opt_method}_all_evaluations_reversed.csv', index=False)
    print_slowly(f"All evaluations saved to output/{RUN_NAME}/{opt_method}_all_evaluations.csv")

    return optimized_formulations

# objective function for BO (single-objective only)
def objective_fcn_BO(model, input_param_names=None, output_scaler = None, all_evaluations=None, **params):
  x = list(params.values())
  y = model.predict(np.array(x).reshape(1, -1))[0]

  # Log evaluation
  reversed_y = output_scaler.inverse_transform([[y]])[0][0]
  entry = {input_param_names[i]: x[i] for i in range(len(x))}
  entry['Predicted_LnRLU'] = reversed_y
  all_evaluations.append(entry)

  return y

# objective function for DA (single-objective only)
def objective_fcn_DA(x, model, input_param_names=None, output_scaler = None, all_evaluations=None):
  y = model.predict(np.array(x).reshape(1, -1))[0]

  # Log evaluation
  reversed_y = output_scaler.inverse_transform([[y]])[0][0]
  entry = {input_param_names[i]: x[i] for i in range(len(x))}
  entry['Predicted_LnRLU'] = reversed_y
  all_evaluations.append(entry)

  return -y


def valid_formulation(formulation, input_param_names, verbose = False):
    """
    Check if the formulation is valid based on predefined criteria.
    """
    # All percentage parameters should be between 0 and 100 
    for i in range(len(formulation)):
        if input_param_names[i] in ['PEG_(Chol+PEG)', '(IL+HL)', 'HL_(IL+HL)','SORT_of_total']:
            if formulation[i] < 0 or formulation[i] > 100:
                if verbose:
                    print("LNP rejected, lipid percentage out of range")
                return False 
        if input_param_names[i] == 'IL_NP_ratio':
            if formulation[i] < 2 or formulation[i] > 12: # limit at 2 and 12 for NP_ratio
                if verbose:
                    print(f"LNP rejected, {formulation[i]} NP ratio out of range")
                return False
    return True

#SHAP importance weighted euclidian distance diversity thresholding
def shap_euc_exclusion(formulation, feature_importance, diversity_threshold,
                       historical_data, current_selection, verbose=False):
    """
    Checks if the proposed formulation is sufficiently diverse from historical and selected points,
    using feature-importance-weighted Euclidean distance.
    """
    # Normalize feature importance
    feat_weights = feature_importance / feature_importance.max()

    # Combine historical and selected data
    combined = historical_data if current_selection.empty else pd.concat([historical_data, current_selection])

    # Compute weighted distances
    form = np.array(formulation, dtype=float)
    for idx, hist in combined.iterrows():
        hist_vals = hist.values.astype(float)
        diffs = hist_vals - form
        weighted_sq = feat_weights.values * diffs**2
        distance = np.sqrt(np.sum(weighted_sq))

        if distance < diversity_threshold:
            if verbose:
                print(f"LNP rejected (diversity = {distance:.4f} < {diversity_threshold})")
            return False

    if verbose:
        print(f"LNP accepted (all distances >= {diversity_threshold})")
    return True


# Shap analysis run - outputs 
def shap_analysis(RUN_NAME, cell_type,pipeline):
    # store models in dictionary 
    models = {}
    input_scalars = {}
    output_scalars = {}

    # model_path = f'../output/{RUN_NAME}/{cell_type}/'
    # with open(f'{model_path}Pipeline_dict.pkl', 'rb') as file:
    #     pipeline = pickle.load(file)
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


# NSGAII: modified Problem class to handle multiple XGBoost models
class FormulationOptimizationProblem(Problem):
    def __init__(self, direction, input_param_names,*xgb_models, pbounds):
        super().__init__(n_var=len(input_param_names), n_obj=len(xgb_models), n_constr=0, xl=pbounds[0], xu=pbounds[1])
        
        self.direction = direction
        self.input_param_names = input_param_names

        for i, model in enumerate(xgb_models, start=1):
            setattr(self, f"xgb_model_y{i}", model)

    def _evaluate(self, X, out, *args, **kwargs):
        predictions = []
        for i, sign in enumerate(self.direction, start=1):
            model = getattr(self, f"xgb_model_y{i}")
            pred = model.predict(X)
            predictions.append(sign * pred)

        out["F"] = np.column_stack(predictions)

# NSGAII: custom Mutation and Crossover classes for adaptive behavior
class AdaptiveMutation(Mutation):
    def __init__(self, base_prob=0.1, eta=20, diversity_threshold=0.1):
        super().__init__()
        self.base_prob = base_prob
        self.eta = eta
        self.diversity_threshold = diversity_threshold

    def _do(self, problem, X, **kwargs):
        diversity = np.mean(np.std(X, axis=0))
        if diversity < self.diversity_threshold:
            prob = self.base_prob * 2  # Increase mutation rate if diversity is low
        else:
            prob = self.base_prob
        return PolynomialMutation(prob=prob, eta=self.eta)._do(problem, X, **kwargs)
class AdaptiveCrossover(Crossover):
    def __init__(self, base_prob=0.9, eta=15, diversity_threshold=0.1):
        super().__init__(2, 2)
        self.base_prob = base_prob
        self.eta = eta
        self.diversity_threshold = diversity_threshold

    def _do(self, problem, X, **kwargs):
        diversity = np.mean(np.std(X, axis=0))
        if diversity < self.diversity_threshold:
            prob = self.base_prob * 0.5  # Decrease crossover rate if diversity is low
        else:
            prob = self.base_prob
        return SimulatedBinaryCrossover(prob=prob, eta=self.eta)._do(problem, X, **kwargs)
    
# Diversity-weighted greedy selection of points and criteria based diversity selection
def greedy_selection(points, n_select, input_param_names, models, input_scalars, output_scalars,
                     cell_type_list, opt_method, feature_importance, training_data, diversity_threshold):

    selector = DiverseValidSelector(
        input_param_names=input_param_names,
        models=models,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
        cell_type_list=cell_type_list,
        feature_importance=feature_importance,
        training_data=training_data,
        diversity_threshold=diversity_threshold,
        opt_method=opt_method
    )

    remaining = points.tolist()

    # Step 1: Select the first valid and diverse point
    for x in remaining:
        if selector.try_add_point(x):
            break
    else:
        raise RuntimeError("No valid initial formulation found.")

    # Step 2: Greedy selection
    eval_count = 0
    while len(selector.optimized_formulations) < n_select:
        eval_count += 1
        if not remaining:
            print(f"No remaining points to evaluate; {len(selector.optimized_formulations)} valid formulations found.")
            break

        # Select the most diverse point from those remaining
        next_point = max(
            remaining,
            key=lambda x: min(np.linalg.norm(np.array(x) - np.array(prev)) for prev in selector.selected_normalized)
        )
        remaining.remove(next_point)
        selector.try_add_point(next_point, verbose=True)

        if eval_count >= 1000:
            print(f"Max (1000) evaluations reached; {len(selector.optimized_formulations)} valid formulations found.")
            break

    return selector.get_optimized_formulations()

#FOR i-optimal samplings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc

#create a gaussian process model trained on the xgb models for LNP predictions
def fit_gp_on_xgb_output(X_train, xgb_model):
    y_pred = xgb_model.predict(X_train)
    kernel = C(1.0) * RBF(length_scale=0.2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_train, y_pred)
    return gp

def i_optimal_search(X_train, X_candidates, X_eval, pred_model, gp, selected):
    min_avg_var = np.inf
    #Test and find best candidate
    for x in X_candidates:
        X_aug = np.vstack([X_train] + selected + [x])
        y_aug = pred_model.predict(X_aug)

        gp_temp = GaussianProcessRegressor(kernel=gp.kernel_, alpha=1e-6, normalize_y=True)
        gp_temp.fit(X_aug, y_aug)

        _, std = gp_temp.predict(X_eval, return_std=True)
        avg_var = np.mean(std**2) ###OBJECTIVE FOR i-optimal sampling

        if avg_var < min_avg_var:
            min_avg_var = avg_var
            best_x = x
    
    return best_x, min_avg_var 


# SELECTION CRITERIA
class DiverseValidSelector:
    def __init__(self, input_param_names, models, input_scalars, output_scalars,
                 cell_type_list, feature_importance, training_data, diversity_threshold, opt_method=""):
        self.input_param_names = input_param_names
        self.models = models
        self.input_scalars = input_scalars
        self.output_scalars = output_scalars
        self.cell_type_list = cell_type_list
        self.feature_importance = feature_importance
        self.training_data = training_data
        self.diversity_threshold = diversity_threshold
        self.opt_method = opt_method

        self.selected_normalized = []
        self.test_selected = pd.DataFrame(columns=input_param_names)
        self.optimized_formulations = []

    def inverse_transform_input(self, x):
        return [
            self.input_scalars[self.cell_type_list[0]][name].inverse_transform([[x[i]]])[0][0]
            for i, name in enumerate(self.input_param_names)
        ]

    def predict_all_outputs(self, x):
        return {
            cell_type: self.output_scalars[cell_type].inverse_transform(
                self.models[cell_type].predict(np.array(x).reshape(1, -1)).reshape(-1, 1)
            )[0][0]
            for cell_type in self.cell_type_list
        }

    def valid_formulation(self, formulation, verbose=False):
        for i, name in enumerate(self.input_param_names):
            value = formulation[i]
            if name in ['PEG_(Chol+PEG)', '(IL+HL)', 'HL_(IL+HL)', 'SORT_of_total']:
                if value < 0 or value > 100:
                    if verbose:
                        print("LNP rejected, lipid percentage out of range")
                    return False
            elif name == 'IL_NP_ratio':
                if value < 2 or value > 12:
                    if verbose:
                        print(f"LNP rejected, {value} NP ratio out of range")
                    return False
        return True

    def shap_euc_exclusion(self, formulation, verbose=False):
        feat_weights = self.feature_importance / self.feature_importance.max()
        combined = self.training_data if self.test_selected.empty else pd.concat([self.training_data, self.test_selected])
        form = np.array(formulation, dtype=float)

        for idx, hist in combined.iterrows():
            hist_vals = hist.values.astype(float)
            diffs = hist_vals - form
            weighted_sq = feat_weights.values * diffs ** 2
            distance = np.sqrt(np.sum(weighted_sq))
            if distance < self.diversity_threshold:
                if verbose:
                    print(f"LNP rejected (diversity = {distance:.4f} < {self.diversity_threshold})")
                return False

        if verbose:
            print(f"LNP accepted (all distances >= {self.diversity_threshold})")
        return True

    def try_add_point(self, x, verbose=False):
        reversed_x = self.inverse_transform_input(x)
        if self.valid_formulation(reversed_x, verbose=verbose) and \
           self.shap_euc_exclusion(x, verbose=verbose):

            y_dict = self.predict_all_outputs(x)
            self.selected_normalized.append(x)
            self.test_selected.loc[len(self.test_selected)] = x

            result_row = {name: val for name, val in zip(self.input_param_names, reversed_x)}
            result_row.update({f"Pred_LnRLU_{k}": v for k, v in y_dict.items()})
            result_row["opt_method"] = self.opt_method
            self.optimized_formulations.append(result_row)

            print_slowly(
                f"Optim. Form. {len(self.optimized_formulations)}: " +
                ', '.join(f"{n}: {v:.3f}" for n, v in zip(self.input_param_names, reversed_x)) +
                " -> " +
                ', '.join(f"Pred. LnRLU_{k}: {v:.3f}" for k, v in y_dict.items())
            )
            return True
        return False

    def get_optimized_formulations(self):
        return pd.DataFrame(self.optimized_formulations)