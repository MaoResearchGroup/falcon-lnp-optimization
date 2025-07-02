# optimization_search.py

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from falcon_engine.utilities import print_slowly
import shap
import pickle
import time

# --- Helper Classes & Functions ---
from .diverse_selector import DiverseValidSelector
from .nsga_components import FormulationOptimizationProblem, AdaptiveCrossover, AdaptiveMutation
import numpy as np
import pandas as pd




class OptimizationSearch:
    def __init__(self, models, input_scalars, output_scalars, training_data, input_param_names,
                 cell_type_list, feature_importance, diversity_threshold, opt_method, norm_bounds):

        self.models = models
        self.input_scalars = input_scalars
        self.output_scalars = output_scalars
        self.training_data = training_data
        self.input_param_names = input_param_names
        self.cell_type_list = cell_type_list
        self.feature_importance = feature_importance
        self.diversity_threshold = diversity_threshold
        self.opt_method = opt_method
        self.norm_bounds = norm_bounds
        self.all_evaluations = []

        self.selector = DiverseValidSelector(
            input_param_names=self.input_param_names,
            models=self.models,
            input_scalars=self.input_scalars,
            output_scalars=self.output_scalars,
            cell_type_list=self.cell_type_list,
            feature_importance=self.feature_importance,
            training_data=self.training_data[self.cell_type_list[0]],
            diversity_threshold=self.diversity_threshold,
            opt_method=self.opt_method
        )

        print("SCALED SUGGESTION BOUNDS", self.norm_bounds)

    def run_i_optimal(self, num_formulations):
        print_slowly("\n--- STARTING i-optimal OPTIMIZATION ---")
        model = self.models[self.cell_type_list[0]]
        X_train = self.training_data[self.cell_type_list[0]]
        gp = self._fit_gp_on_model(X_train, model)

        bounds = self.norm_bounds
        sampler = qmc.LatinHypercube(len(bounds))
        X_candidates = qmc.scale(sampler.random(1000), [b[0] for b in bounds], [b[1] for b in bounds])
        X_eval = qmc.scale(sampler.random(500), [b[0] for b in bounds], [b[1] for b in bounds])

        count = 0
        while len(self.selector.optimized_formulations) < num_formulations and len(X_candidates) > 0:
            count += 1
            if count >= 100:
                print(f"Max (100) searches reached; {len(self.selector.optimized_formulations)} valid found.")
                break

            suggested_x, min_avg_var = self._i_optimal_search(X_train, X_candidates, X_eval, model, gp)
            X_candidates = np.delete(X_candidates, np.where((X_candidates == suggested_x).all(axis=1))[0], axis=0)
            self.selector.try_add_point(suggested_x, verbose=True)

        return self.selector.get_optimized_formulations()

    def run_bayesian(self, num_formulations):
        print_slowly("\n--- STARTING BAYESIAN OPTIMIZATION ---")
        # pbounds = {f'param{i}': self.norm_bounds for i in range(1, len(self.input_param_names)+1)}
        # print({f'param{i}': self.norm_bounds for i in range(1, len(self.input_param_names)+1)})
        # pbounds = self.norm_bounds

        # Convert self.norm_bounds to a dictionary format expected by BayesianOptimization
        pbounds = {f'param{i+1}': self.norm_bounds[i] for i in range(len(self.input_param_names))}
        print("Parameter bounds (scaled):", pbounds)
        while len(self.selector.optimized_formulations) < num_formulations:
            optimizer = BayesianOptimization(
                f=lambda **params: self._objective_fcn_BO(self.all_evaluations, **params),
                pbounds=pbounds,
                verbose=2
            )
            optimizer.maximize(init_points=10, n_iter=50)
            maximal_x = np.array(list(optimizer.max['params'].values()))
            self.selector.try_add_point(maximal_x, verbose=True)

        return self.selector.get_optimized_formulations()

    def run_dual_annealing(self, num_formulations):
        print_slowly("\n--- STARTING DUAL ANNEALING OPTIMIZATION ---")
        bounds = self.norm_bounds

        while len(self.selector.optimized_formulations) < num_formulations:
            result = dual_annealing(
                func=self._objective_fcn_DA(self.all_evaluations),
                bounds=bounds,
                maxiter=100,
                initial_temp=20000, 
                visit=2.8
            )
            maximal_x = result.x
            self.selector.try_add_point(maximal_x, verbose=True)

        return self.selector.get_optimized_formulations()
    
    def run_nsga2(self, num_formulations, max_cell_targets, min_cell_targets):
        print_slowly("\n--- STARTING NSGA-II OPTIMIZATION ---")

        direction = [-1] * len(max_cell_targets) + [1] * len(min_cell_targets)
        ordered_models = [self.models[cell] for cell in self.cell_type_list]
        problem = FormulationOptimizationProblem(direction, self.input_param_names, *ordered_models, pbounds=self.norm_bounds)

        seed = 1
        max_seeds = 20
        while seed <= max_seeds:

            if len(self.selector.optimized_formulations) >= num_formulations:
                break

            print(f"\n>>> Running NSGA-II with seed {seed}")
            algorithm = NSGA2(
                pop_size=500,
                sampling=LHS(),
                crossover=AdaptiveCrossover(base_prob=0.9, eta=15),
                mutation=AdaptiveMutation(base_prob=0.1, eta=20),
                eliminate_duplicates=True
            )

            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', 20),
                seed=seed,
                save_history=False,
                verbose=False
            )

            front = NonDominatedSorting().do(res.pop.get("F"), only_non_dominated_front=True)
            pareto_solutions = res.pop.get("X")[front]

            self._greedy_selection(pareto_solutions, num_formulations)
            seed += 1

        print(f"Finished NSGA-II: total selected = {len(self.selector.optimized_formulations)}")
        return self.selector.get_optimized_formulations()


    def _greedy_selection(self, points, n_select):
        remaining = points.tolist()
        print(remaining)

        for x in remaining:
            if self.selector.try_add_point(x, verbose = True):
                break
        else:
            print(f"No valid initial formulation found continuing to next seed")
            return

        eval_count = 0
        while len(self.selector.optimized_formulations) < n_select:
            eval_count += 1
            if not remaining:
                print(f"No remaining points to evaluate; {len(self.selector.optimized_formulations)} valid formulations found.")
                break

            next_point = max(
                remaining,
                key=lambda x: min(np.linalg.norm(np.array(x) - np.array(prev)) for prev in self.selector.selected_normalized)
            )
            remaining.remove(next_point)
            self.selector.try_add_point(next_point, verbose=True)

            if eval_count >= 1000:
                print(f"Max (1000) evaluations reached; {len(self.selector.optimized_formulations)} valid formulations found.")
                break

    # --- Internal Utilities ---
    def _fit_gp_on_model(self, X_train, model):
        y_pred = model.predict(X_train)
        kernel = C(1.0) * RBF(length_scale=0.2)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        gp.fit(X_train, y_pred)
        return gp

    def _i_optimal_search(self, X_train, X_candidates, X_eval, pred_model, gp):
        min_avg_var = np.inf
        best_x = None
        for x in X_candidates:
            X_aug = np.vstack([X_train] + self.selector.selected_normalized + [x])
            y_aug = pred_model.predict(X_aug)

            gp_temp = GaussianProcessRegressor(kernel=gp.kernel_, alpha=1e-6, normalize_y=True)
            gp_temp.fit(X_aug, y_aug)

            _, std = gp_temp.predict(X_eval, return_std=True)
            avg_var = np.mean(std**2)
            if avg_var < min_avg_var:
                min_avg_var = avg_var
                best_x = x
        return best_x, min_avg_var
    # def _nsga2_search(self, num_formulations, max_cell_targets, min_cell_targets, seed):
    #     print_slowly("\n--- STARTING NSGA-II OPTIMIZATION ---")

    #     direction = [-1] * len(max_cell_targets) + [1] * len(min_cell_targets)
    #     ordered_models = [self.models[cell] for cell in self.cell_type_list]

    #     problem = FormulationOptimizationProblem(direction, self.input_param_names, *ordered_models, pbounds=self.norm_bounds)
    #     algorithm = NSGA2(
    #         pop_size=500,
    #         sampling=LHS(),
    #         crossover=AdaptiveCrossover(base_prob=0.9, eta=15),
    #         mutation=AdaptiveMutation(base_prob=0.1, eta=20),
    #         eliminate_duplicates=True
    #     )

    #     res = minimize(problem, algorithm, termination=('n_gen', 250), seed=seed, save_history=True, verbose=True)

    #     front = NonDominatedSorting().do(res.pop.get("F"), only_non_dominated_front=True)
    #     pareto_solutions = res.pop.get("X")[front]

    #     return self._greedy_selection(pareto_solutions, num_formulations)

    def _objective_fcn_BO(self, all_evaluations, **params):
        x = list(params.values())
        model = self.models[self.cell_type_list[0]]
        y = model.predict(np.array(x).reshape(1, -1))[0]

        reversed_y = self.output_scalars[self.cell_type_list[0]].inverse_transform([[y]])[0][0]
        entry = {self.input_param_names[i]: x[i] for i in range(len(x))}
        entry['Predicted_LnRLU'] = reversed_y
        all_evaluations.append(entry)

        return y

    def _objective_fcn_DA(self, all_evaluations):
        def wrapped(x):
            model = self.models[self.cell_type_list[0]]
            y = model.predict(np.array(x).reshape(1, -1))[0]

            reversed_y = self.output_scalars[self.cell_type_list[0]].inverse_transform([[y]])[0][0]
            entry = {self.input_param_names[i]: x[i] for i in range(len(x))}
            entry['Predicted_LnRLU'] = reversed_y
            all_evaluations.append(entry)

            return -y
        return wrapped

