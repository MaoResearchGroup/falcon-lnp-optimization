# nsga_components.py

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

class FormulationOptimizationProblem(Problem):
    '''
    NSGA-II Components for Multi-Objective Formulation Optimization. 
    Defines custom problem and operator classes used in FALCON pipeline. 
    '''
    def __init__(self, direction, input_param_names, *xgb_models, pbounds):
        # Unpack parameter-specific bounds
        xl_list = np.array([low for (low, high) in pbounds])
        xu_list = np.array([high for (low, high) in pbounds])

        super().__init__(
            n_var=len(input_param_names),
            n_obj=len(xgb_models),
            n_constr=0,
            xl=xl_list,
            xu=xu_list
        )

        self.direction = direction
        self.input_param_names = input_param_names
        self.models = list(xgb_models)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            F.append(self.direction[i] * pred)
        out["F"] = np.column_stack(F)


class AdaptiveMutation(Mutation):
    def __init__(self, base_prob=0.1, eta=20, diversity_threshold=0.1):
        super().__init__()
        self.base_prob = base_prob
        self.eta = eta
        self.diversity_threshold = diversity_threshold

    def _do(self, problem, X, **kwargs):
        diversity = np.mean(np.std(X, axis=0))
        prob = self.base_prob * 2 if diversity < self.diversity_threshold else self.base_prob
        return PolynomialMutation(prob=prob, eta=self.eta)._do(problem, X, **kwargs)


class AdaptiveCrossover(Crossover):
    def __init__(self, base_prob=0.9, eta=15, diversity_threshold=0.1):
        super().__init__(2, 2)
        self.base_prob = base_prob
        self.eta = eta
        self.diversity_threshold = diversity_threshold

    def _do(self, problem, X, **kwargs):
        diversity = np.mean(np.std(X, axis=0))
        prob = self.base_prob * 0.5 if diversity < self.diversity_threshold else self.base_prob
        return SimulatedBinaryCrossover(prob=prob, eta=self.eta)._do(problem, X, **kwargs)
