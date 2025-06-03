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

def run_optimization_pipeline(opt_method, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME):
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
    cell_type_list = MAX_cell_targets + MIN_cell_targets
    models = {}
    input_scalars = {}
    output_scalars = {}

    for cell_type in cell_type_list:
        model_path = f'output/{RUN_NAME}/{cell_type}/'
        with open(f'{model_path}Pipeline_dict.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        models[cell_type] = pipeline['Model_Selection']['Best_Model']['Model']
        output_scalars[cell_type] = pipeline['Data_preprocessing']['Output_Scaler']
        input_scalars[cell_type] = pipeline['Data_preprocessing']['Scalers']
    input_param_names = pipeline['Data_preprocessing']['Input_Params']

    if opt_method == 'DA': 
        print_slowly(f"\n\n--- STARTING DUAL ANNEALING OPTIMIZATION for high {cell_type_list [0]} transfection ---")
        #set search bounds for optimization 
        bounds = [(0, 1.2) for _ in range(len(input_param_names))] # expanded parameter bounds
        while(len(optimized_formulations) < num_formulations):
            result = dual_annealing(
                func=objective_fcn_DA,
                bounds=bounds,
                args=(models[cell_type_list[0]], input_param_names, output_scalars[cell_type_list[0]], all_evaluations),
                initial_temp=20000, 
                visit = 2.8
            )
            print(f"Dual Annealing Run Result: {result}")
            maximal_x, maximal_y = result.x, -result.fun
            reversed_x = [input_scalars[cell_type_list[0]][input_param_names[i]].inverse_transform([[maximal_x[i]]])[0][0] for i in range(len(maximal_x))]
            reversed_y = output_scalars[cell_type_list[0]].inverse_transform(np.array(maximal_y).reshape(-1, 1))[0][0]
            if (valid_formulation(reversed_x, input_param_names)): 
                optimized_formulations.append((reversed_x, reversed_y, opt_method))
                #print the optimized formulation
                formatted_params = ", ".join(
                    f"{name}: {value:.3f}" for name, value in zip(input_param_names, reversed_x)
                )
                print_slowly(f"Optimized Formulation {len(optimized_formulations)}: {formatted_params}, Predicted RLU: {reversed_y}") 
            else: 
                print_slowly(f"Invalid formulation found: {reversed_x}, skipping...")
    elif opt_method == 'BO':
        print_slowly(f"\n\n--- STARTING BAYESIAN OPTIMIZATION for high {cell_type_list [0]} transfection ---")
        pbounds = {f'param{i}': (0, 1.2) for i in range(1, len(input_param_names)+1)} #expanded parameter bounds 
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
            maximal_x, maximal_y = np.array(list(optimizer.max['params'].values())), [optimizer.max['target']] 
            reversed_x = [input_scalars[cell_type_list[0]][input_param_names[i]].inverse_transform([[maximal_x[i]]])[0][0] for i in range(len(maximal_x))]
            reversed_y = output_scalars[cell_type_list[0]].inverse_transform(np.array(maximal_y).reshape(-1, 1))[0][0]
            if (valid_formulation(reversed_x, input_param_names)): 
                optimized_formulations.append((reversed_x, reversed_y, opt_method))
                #print the optimized formulation
                formatted_params = ", ".join(
                    f"{name}: {value:.3f}" for name, value in zip(input_param_names, reversed_x)
                )
                print_slowly(f"Optimized Formulation {len(optimized_formulations)}: {formatted_params}, Predicted RLU: {reversed_y}") 
            else: 
                print_slowly(f"Invalid formulation found: {reversed_x}, skipping...")
    elif opt_method == 'NSGAII':
        print_slowly(f"\n\n--- STARTING NSGA-II OPTIMIZATION for high {MAX_cell_targets} and low {MIN_cell_targets} transfection ---")
        sampling = LHS()  # Latin Hypercube Sampling
        crossover = AdaptiveCrossover(base_prob=0.9, eta=15)
        mutation = AdaptiveMutation(base_prob=0.1, eta=20)
        direction = [-1] * len(MAX_cell_targets) + [1] * len(MIN_cell_targets)
        ordered_models = [models[cell] for cell in cell_type_list]
        problem = FormulationOptimizationProblem(direction, input_param_names, *ordered_models)
        algorithm = NSGA2(pop_size=500, sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=True)
        # Run the optimization
        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', 200), # number of generations temporary 
            seed=1,
            save_history=True,
            verbose=True
        )
         # print completion 
        print_slowly("NSGA-II optimization completed, saving all evaluations...")

        # save all the evlaluations 
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

        # Extract the best solutions (pareto optimal solutions)
        print_slowly("Extracting Pareto optimal solutions...")
        pop = res.pop.get("X")
        obj = res.pop.get("F")
        nds = NonDominatedSorting()
        front = nds.do(obj, only_non_dominated_front=True)
        pareto_solutions = pop[front]

        optimized_formulations = greedy_selection(pareto_solutions, num_formulations, input_param_names, models, input_scalars, output_scalars, cell_type_list, opt_method)
      
    print_slowly("\n\n--- %s minutes for OPTIMIZED FORMULATION GENERATION ---" % ((time.time() - start_time)/60))

    print_slowly("\n\n--- EXPORTING ALL EVALUATIONS to csv ---")
    # Save all evaluations to a CSV file
    evaluations_df = pd.DataFrame(all_evaluations)
    evaluations_df.to_csv(f'output/{RUN_NAME}/{opt_method}_all_evaluations.csv', index=False)
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

def valid_formulation(formulation, input_param_names):
    """
    Check if the formulation is valid based on predefined criteria.
    """
    # All percentage parameters should be between 0 and 100 
    for i in range(len(formulation)):
        if input_param_names[i] in ['PEG_PEG+Chol', 'IL+HL', 'HL_IL+HL']:
            if formulation[i] < 0 or formulation[i] > 100:
                return False 
        if input_param_names[i] == 'NP_ratio':
            if formulation[i] < 0 or formulation[i] > 25: # cap at 25 for NP_ratio
                return False
    return True

# NSGAII: modified Problem class to handle multiple XGBoost models
class FormulationOptimizationProblem(Problem):
    def __init__(self, direction, input_param_names,*xgb_models):
        super().__init__(n_var=len(input_param_names), n_obj=len(xgb_models), n_constr=0, xl=0, xu=1.2)
        
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
    
# Diversity-weighted greedy selection of points
def greedy_selection(points, n_select, input_param_names, models, input_scalars, output_scalars, cell_type_list, opt_method):
    selected_normalized = []        # For diversity calculation
    optimized_formulations = []     # Final output: (reversed_x, y_dict)
    remaining = points.tolist()

    def predict_all_outputs(x):
        """Predict outputs from all models and inverse-transform them."""
        y_dict = {}
        for cell_type in cell_type_list:
            model = models[cell_type]
            scaler = output_scalars[cell_type]
            y_pred = model.predict(np.array(x).reshape(1, -1))
            y_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))[0][0]
            y_dict[cell_type] = y_inv
        return y_dict

    def inverse_transform_input(x):
        """Convert normalized input to real-world values using first cell type's scalers."""
        return [
            input_scalars[cell_type_list[0]][name].inverse_transform([[x[i]]])[0][0]
            for i, name in enumerate(input_param_names)
        ]

    # Step 1: Pick the first valid point
    for x in remaining:
        reversed_x = inverse_transform_input(x)
        if valid_formulation(reversed_x, input_param_names):
            y_dict = predict_all_outputs(x)
            selected_normalized.append(x)
            optimized_formulations.append((reversed_x, y_dict, opt_method))
            remaining.remove(x)

            print_slowly(
                f"Optim. Form. {len(optimized_formulations)}: "
                + ', '.join(f"{n}: {v:.3f}" for n, v in zip(input_param_names, reversed_x))
                + " -> "
                + ', '.join(f"Pred. LnRLU_{k}: {v:.3f}" for k, v in y_dict.items())
            )
            break
    else:
        raise RuntimeError("No valid initial formulation found.")

    # Step 2: Greedy selection of remaining diverse points
    while len(optimized_formulations) < n_select:
        next_point = max(
            remaining,
            key=lambda x: min(np.linalg.norm(np.array(x) - np.array(prev_x)) for prev_x in selected_normalized)
        )
        reversed_x = inverse_transform_input(next_point)
        if valid_formulation(reversed_x, input_param_names):
            y_dict = predict_all_outputs(next_point)
            selected_normalized.append(next_point)
            optimized_formulations.append((reversed_x, y_dict, opt_method))
            remaining.remove(next_point)
            print_slowly(
                f"Optim. Form. {len(optimized_formulations)}: "
                + ', '.join(f"{n}: {v:.3f}" for n, v in zip(input_param_names, reversed_x))
                + " -> "
                + ', '.join(f"Pred. LnRLU_{k}: {v:.3f}" for k, v in y_dict.items())
            )
        else:
            print_slowly(f"Invalid formulation found: {next_point}, skipping...")

    return optimized_formulations


