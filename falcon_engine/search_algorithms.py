# optimization_search.py

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.lhs import LHS
from falcon_engine.utilities import print_slowly

# --- Helper Classes & Functions ---
from .diverse_selector import DiverseValidSelector
from .nsga_components import FormulationOptimizationProblem, AdaptiveCrossover, AdaptiveMutation
import numpy as np
import pandas as pd

class OptimizationSearch:
    '''
    Central optimization engine for FALCON formulation discovery. 
    Executes optimization using selected search strategy, logs surrogate evaluations and tracks pareto front. 
    Applies feasibility + diversity filterign via DiverseValidselector class. 
    Exports optimization results for downstram analysis. 
    '''
    def __init__(self, models, input_scalars, output_scalars, training_data, input_param_names,
                 cell_type_list, feature_importance, diversity_threshold, opt_method, norm_bounds, RUN_NAME):

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
        self.RUN_NAME = RUN_NAME

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

    # I-optimal sampling strategy where uncertainty is updated with each iteration, and search bounds are within training data
    # Use GP surrogate to estimate uncertainty of predictions, and select candidates that minimize average variance across the design space for ALL cell type models 
    def run_i_optimal(self, num_formulations):
        self.selector.opt_method = 'i-optimal'
        print_slowly("\n--- STARTING i-optimal OPTIMIZATION ---")

        model = self.models[self.cell_type_list[0]]
        # Start with original training data
        X_train = self.training_data[self.cell_type_list[0]].copy()
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        margin = 0.05 * (X_max - X_min) # small expanison margin
        lb = X_min - margin
        ub = X_max + margin
        #Clip to global (physical) bounds
        global_lb = np.array([b[0] for b in self.norm_bounds])
        global_ub = np.array([b[1] for b in self.norm_bounds])

        lb = np.maximum(lb, global_lb)
        ub = np.minimum(ub, global_ub)
        # ------------------------------------------------
        sampler = qmc.LatinHypercube(X_train.shape[1])
        X_candidates = qmc.scale(sampler.random(1000),lb,ub)
        X_eval = qmc.scale(sampler.random(500),lb,ub)
        count = 0
        local_tally = 0
        while local_tally < num_formulations and len(X_candidates) > 0:
            count += 1
            if count >= 100:
                print(f"Max (100) searches reached; {len(self.selector.optimized_formulations)} valid found.")
                break
            # Fit GP once per cell type (INCLUDING previously selected points)
            gps = {}
            base_eval_var = {}

            for cell in self.cell_type_list:

                # include sequentially selected points
                X_train_cell = np.vstack(
                    [self.training_data[cell]]
                    + self.selector.selected_normalized
                )

                model = self.models[cell]
                gp = self._fit_gp_on_model(X_train_cell, model)

                gps[cell] = gp

                # Precompute base variance over evaluation grid
                _, std_eval = gp.predict(X_eval, return_std=True)
                base_eval_var[cell] = std_eval ** 2

            # Fast closed-form I-optimal search
            suggested_x, min_total_avg_var = self._i_optimal_search_multi_fast(
                X_candidates,
                X_eval,
                gps,
                base_eval_var
            )
            # Remove from candidate pool
            X_candidates = np.delete(
                X_candidates,
                np.where((X_candidates == suggested_x).all(axis=1))[0],
                axis=0
            )
            accepted = self.selector.try_add_point(suggested_x, verbose=True)
            # Only update training set if accepted
            if accepted:
                X_train = np.vstack([X_train, suggested_x])
                local_tally += 1
        return pd.DataFrame(self.selector.optimized_formulations[-num_formulations:])
    
    def run_bayesian(self, num_formulations):
        self.selector.opt_method = 'BO'
        print_slowly("\n--- STARTING BAYESIAN OPTIMIZATION ---")

        # Convert self.norm_bounds to a dictionary format expected by BayesianOptimization
        pbounds = {f'param{i+1}': self.norm_bounds[i] for i in range(len(self.input_param_names))}

        #set bounds as the minimum of manual pbounds (above) or (-0.2, 1.2) in normalized space (intersecting points)
        for key in pbounds:
            pbounds[key] = (max(pbounds[key][0], -0.2), min(pbounds[key][1], 1.2))

        print("Parameter bounds (scaled):", pbounds)
        local_tally = 0
        while local_tally < num_formulations:
            optimizer = BayesianOptimization(
                f=lambda **params: self._objective_fcn_BO(**params),
                pbounds=pbounds,
                verbose=2
            )
            optimizer.maximize(init_points=10, n_iter=50)
            maximal_x = np.array(list(optimizer.max['params'].values()))
            accepted = self.selector.try_add_point(maximal_x, verbose=True)
            if accepted:
                local_tally += 1

        return pd.DataFrame(self.selector.optimized_formulations[-num_formulations:])

    def run_dual_annealing(self, num_formulations):
        self.selector.opt_method = 'DA'
        print_slowly("\n--- STARTING DUAL ANNEALING OPTIMIZATION ---")

        bounds = [
            (max(lower, -0.2), min(upper, 1.2))
            for lower, upper in self.norm_bounds
        ]
        
        local_tally = 0
        while local_tally < num_formulations:
            result = dual_annealing(
                func=self._objective_fcn_DA(),
                bounds=bounds,
                maxiter=100, #simplify for demo
                initial_temp=20000, 
                visit=2.8
            )
            maximal_x = result.x
            accepted = self.selector.try_add_point(maximal_x, verbose=True)
            if accepted:
                local_tally += 1
        return pd.DataFrame(self.selector.optimized_formulations[-num_formulations:])
    
    def run_nsga2(self, num_formulations, max_cell_targets, min_cell_targets):
        self.selector.opt_method = 'NSGAII'
        print_slowly("\n--- STARTING NSGA-II OPTIMIZATION ---")

        clipped_bounds = [
        (max(l, -0.2), min(u, 1.2))
        for l, u in self.norm_bounds
        ]

        direction = [-1] * len(max_cell_targets) + [1] * len(min_cell_targets)
        ordered_models = [self.models[cell] for cell in self.cell_type_list]
        problem = FormulationOptimizationProblem(direction, self.input_param_names, *ordered_models, pbounds=clipped_bounds) 

        seed = 1
        max_seeds = 10
        local_tally = 0

        all_histories = []
        all_pareto_F = []
        all_pareto_X = []

        while seed <= max_seeds:

            if local_tally >= num_formulations:
                break

            print(f"\n>>> Running NSGA-II with seed {seed}")
            algorithm = NSGA2(
                pop_size=1000, 
                sampling=LHS(),
                crossover=AdaptiveCrossover(base_prob=0.9, eta=15),
                mutation=AdaptiveMutation(base_prob=0.1, eta=20),
                eliminate_duplicates=True
            )

            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', 250), # simplify for demo
                seed=seed,
                save_history=True,
                verbose=True
            )

            all_histories.append(res.history)
            all_pareto_F.append(res.F)
            all_pareto_X.append(res.X)

            front = NonDominatedSorting().do(res.pop.get("F"), only_non_dominated_front=True)
            pareto_solutions = res.pop.get("X")[front]

            local_tally = self._greedy_selection(pareto_solutions, num_formulations, local_tally)
            seed += 1

        print(f"Finished NSGA-II: total selected = {len(self.selector.optimized_formulations)}")
        print_slowly("Exporting results...")

        all_F = np.vstack([
            np.array([ind.F for ind in gen.pop])
            for history in all_histories
            for gen in history
        ])

        all_X = np.vstack([
            gen.pop.get("X")
            for history in all_histories
            for gen in history
        ])

        # Combine final Pareto fronts from each seed
        combined_pareto_F = np.vstack(all_pareto_F)
        combined_pareto_X = np.vstack(all_pareto_X)

        # Recompute global non-dominated front
        nds = NonDominatedSorting()
        front = nds.do(combined_pareto_F, only_non_dominated_front=True)

        global_pareto_F = combined_pareto_F[front]
        global_pareto_X = combined_pareto_X[front]

        export_dict = {
            'pareto_F': global_pareto_F,
            'all_F': all_F,
            'cell_type_names': self.cell_type_list,
            'pareto_X': global_pareto_X,
            'direction': direction
        }

        with open(f'output/{self.RUN_NAME}/NSGAII_results.pkl', "wb") as f:
            pickle.dump(export_dict, f)

        for cell_type in self.cell_type_list:
            model = self.models[cell_type]
            output_scaler = self.output_scalars[cell_type]

            y_preds = model.predict(all_X)
            y_reversed = output_scaler.inverse_transform(
                y_preds.reshape(-1, 1)
            ).flatten()

            for i, x in enumerate(all_X):
                if i >= len(self.all_evaluations):
                    self.all_evaluations.append(
                        {self.input_param_names[j]: x[j] for j in range(len(x))}
                    )

                self.all_evaluations[i][f'Pred_nLnLE_{cell_type}'] = y_reversed[i]

        return pd.DataFrame(self.selector.optimized_formulations[-num_formulations:])

    def _greedy_selection(self, points, n_select, local_tally):
        remaining = [np.array(x).tolist() for x in points]

        for x in remaining:
            if self.selector.try_add_point(x, verbose = True):
                local_tally += 1
                break
        else:
            print(f"No valid initial formulation found continuing to next seed")
            return local_tally

        eval_count = 0
        while local_tally < n_select:
            eval_count += 1
            if not remaining:
                print(f"No remaining points to evaluate; {len(self.selector.optimized_formulations)} valid formulations found.")
                break

            next_point = max(
                remaining,
                key=lambda x: min(np.linalg.norm(np.array(x) - np.array(prev)) for prev in self.selector.selected_normalized)
            )
            remaining.remove(next_point)
            accepted = self.selector.try_add_point(next_point, verbose=True)
            if accepted:
                local_tally += 1

            if eval_count >= 1000:
                print(f"Max (1000) evaluations reached; {len(self.selector.optimized_formulations)} valid formulations found.")
                break
        return local_tally

    # --- Internal Utilities ---
    def _fit_gp_on_model(self, X_train, model):
        y_pred = model.predict(X_train)
        kernel = C(1.0) * RBF(length_scale=0.2)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        gp.fit(X_train, y_pred)
        return gp
    
    def _i_optimal_search_multi_fast(self, X_candidates, X_eval, gps, base_eval_var):

        min_total_avg_var = np.inf
        best_x = None

        for x in X_candidates:

            self.log_evaluation(x)

            total_avg_var = 0
            x_reshaped = x.reshape(1, -1)

            for cell, gp in gps.items():

                kernel = gp.kernel_
                noise = gp.alpha if np.isscalar(gp.alpha) else np.mean(gp.alpha)

                # variance at candidate
                _, std_star = gp.predict(x_reshaped, return_std=True)
                var_star = std_star[0] ** 2

                # kernel between eval points and candidate
                k_eval_star = kernel(X_eval, x_reshaped).flatten()

                # closed-form updated variance
                var_new = base_eval_var[cell].flatten() - (
                    (k_eval_star ** 2) / (var_star + noise)
                )

                total_avg_var += np.mean(var_new)

            if total_avg_var < min_total_avg_var:
                min_total_avg_var = total_avg_var
                best_x = x

        return best_x, min_total_avg_var
    
    def _objective_fcn_BO(self, **params):
        x = list(params.values())
        model = self.models[self.cell_type_list[0]]
        y = model.predict(np.array(x).reshape(1, -1))[0]

        self.log_evaluation(x)

        return y

    def _objective_fcn_DA(self):
        def wrapped(x):
            model = self.models[self.cell_type_list[0]]
            y = model.predict(np.array(x).reshape(1, -1))[0]

            self.log_evaluation(x)

            return -y
        return wrapped

    def log_evaluation(self, x):
        entry = {name: x[i] for i, name in enumerate(self.input_param_names)}

        for cell_type in self.cell_type_list:
            y = self.models[cell_type].predict(np.array(x).reshape(1, -1))
            y_reversed = self.output_scalars[cell_type]\
                .inverse_transform(y.reshape(-1, 1))[0][0]
            entry[f"Pred_nLnLE_{cell_type}"] = y_reversed

        self.all_evaluations.append(entry)