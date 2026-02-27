import numpy as np
import pandas as pd
from falcon_engine.utilities import print_slowly

class DiverseValidSelector:
    '''
    Class to manage selection of optimized formulations with diversity and validity constraints. 
    Acts as post-professing filter for candidate formulations proposed by optimization algorithms. 
    '''
    def __init__(self, input_param_names, models, input_scalars, output_scalars,
                 cell_type_list, feature_importance, training_data, diversity_threshold, opt_method=""):
        self.input_param_names = input_param_names
        self.models = models
        self.input_scalars = input_scalars
        self.output_scalars = output_scalars
        self.cell_type_list = cell_type_list
        self.feature_importance = feature_importance # this is the maximum feature importance across cell types for the feature of interest
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

    def valid_formulation(self, formulation, verbose=False): #  check that params that are supposed to be percentages
        for i, name in enumerate(self.input_param_names):
            value = formulation[i]
            if name in ['PEG_PEG+Chol', 'IL+HL', 'HL_IL+HL']: 
                if value < 0 or value > 100:
                    if verbose:
                        print("LNP rejected, lipid percentage out of range")
                    return False
        return True

    #compute euclidean distance from proposed formulation to all existing formulations, with each feature weighted by their absolute SHAP importance
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