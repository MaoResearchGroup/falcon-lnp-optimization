
from falcon_engine.utilities import extract_training_data, init_pipeline, save_pipeline, startup_banner
from falcon_engine.run_Model_Selection import run_Model_Selection 
from falcon_engine.run_optimization import run_optimization_pipeline
from falcon_engine import learning_curve
import pandas as pd
import pickle 
from falcon_engine.utilities import print_slowly
from falcon_engine.run_mantis_compiled_formatter import run_mantis_formatter_pipeline

"""
run_FALCON script

- Main script for end-to-end execution of FALCON computational pipeline: 
  - Part 1 (surrogate model training): trains XGBoost models to predict LNP transfection in specified cell types
  - Part 2 (optimization): leverages ML-guided algorithms to suggest LNP compositions for experimental testing.
  - Part 3 (optional add on): formats optimized compositions for MANTIS liquid handler input 
  - note: each part can be run independently (previous runs exist in output folder) by setting corresponding flags to True or False. 
- To run: follow 4 steps in main function to configure pipeline, then run script
  - Adjustable parameters:
    - Optimization method: DA, BO, or NSGAII (opt_method)
    - Number of formulations to generate (num_formulations)
    - Cell types for maximization and minimization objectives (MAX_cell_targets, MIN_cell_targets)
    - Dataset name and path for training data (DATASET_NAME)
    - Run name for saving trained models and optimized formulations (RUN_NAME)
    - input_params for model training (input_param_names)
- Output files exported to RUN_NAME folder (saved trained models, optimized formulations, model diagnostics, etc.)

"""

def main():

  ############### STEP 1: SEARCH CONFIGURATION #########################
  opt_methods = ["DA", "i-optimal"] # DA, BO, NSGAII, i-optimal
  num_formulations = 5 #Default = 12 

  raw_suggestion_bounds = {
        'NP_ratio': (4,10),
        'PEG_(Chol+PEG)': (2, 10),
        '(IL+HL)':(20,90),
        'HL_(IL+HL)':(5,70),
        # 'SORT_of_total': (0,80)
        }
  
  # raw_suggestion_bounds = {
  #       'IL_Mol_RNA': (6,60),
  #       'HL_Mol_RNA': (0, 60),
  #       'Chol_Mol_RNA':(5,100),
  #       'PEG_Mol_RNA':(0, 10),
  #       # 'SORT_of_total': (0,80)
  #       }
    
  diversity_threshold = 0.1 #diversity threshold: how diverse do you want your parameters to be? ([0,1], 1 is more diverse)
  ############### STEP 2: CELL TYPES AND OBJECTIVE CONFIGURATION #######
  # ex. cell types used in manuscript ['RAMOS','DC','3T3','C2C12'] 
  MAX_cell_targets = ['T']
  MIN_cell_targets = [] # set as empty list if no minimization is desired (not '') 
  #MIN_cell_targets = ['DC','3T3','C2C12']

  #model training will be done for each cell type in this list
  #DA and BO will only use first cell type in this list for maximization, NSGAII will use all cell types
  cell_type_list = MAX_cell_targets + MIN_cell_targets 

  ################ STEP 3: LOAD AND SAVE PATH CONFIGURATION #############
  RUN_NAME = "tATLAS_final_kNN" #Give a name for run folder to save any trained models
  DATASET_NAMES = {"T":'tATLAS_Final_kNN',
                   'h19RAMOS': 'bATLAS_MasterCompiled_reversed',
                   'DeltaRAMOS': 'bATLAS_MasterCompiled_reversed'} 
  #Name of the csv file, used to extract training data}

  ################ STEP 4: PIPELINE COMPONENTS CONFIGURATION #############
  run_model_training = False  # set true unless model is already trained and saved in output folder
  run_optimization = True # set true unless de novo formulation generation is not desired 
  run_mantis_formatter = False # set true if you want to format the optimized formulations for MANTIS (liquid handler) input

  ########################################################################
   #Path to the dataset to be used for training

  # Input_Params (features to be used for model training and prediction) 
  # remove for now - 'NP_ratio',
  input_param_names = [ 'NP_ratio',
                        '(IL+HL)',
                        'HL_(IL+HL)',
                        'PEG_(Chol+PEG)']


  # #MOLAR AMOUNTS PARAMETERS 
  # input_param_names = [ 'IL_Mol_RNA',
  #                       'HL_Mol_RNA',
  #                       'Chol_Mol_RNA',
  #                       'PEG_Mol_RNA']
  

  if run_model_training == True:  
    LnRLU_floor = -100 #cutoff below which LnRLU values are considered 0
    for c in cell_type_list:   #Loop through model training for each cell type of interest
      pipeline_path = f'output/{RUN_NAME}/{c}/Pipeline_dict.pkl'
      data_file_path = f'datasets/{DATASET_NAMES[c]}.xlsx'
      #Initialize new model pipeline
      pipeline_dict = init_pipeline(pipeline_path = pipeline_path,
                                      RUN_NAME=RUN_NAME,
                                      input_param_names=input_param_names,
                                      cell = c,
                                      model_list=['XGB'], #List of models to be trained, currently only XGB is supported for surrogate modeling
                                      param_type = 'percent',
                                      data_file_path=data_file_path,
                                      prefix='Fold_',
                                      RLU_floor=LnRLU_floor,
                                      N_CV=5)
      pipeline_dict, _, _, _= extract_training_data(pipeline_dict) 
      pipeline_dict, _, _, _ = run_Model_Selection(pipeline_dict)
      # pipeline_dict = learning_curve.get_learning_curve(pipeline_dict, refined = False)
      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, step = 'FINAL SAVE')  
  
  if run_optimization == True:
    optimized_formulations = pd.DataFrame()
    for opt_method in opt_methods:
      new_suggestions = run_optimization_pipeline(opt_method, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME, diversity_threshold,
                                                  raw_suggestion_bounds)

      optimized_formulations = pd.concat([optimized_formulations, new_suggestions], ignore_index=True)



    print(optimized_formulations)

    #export optimized_formulations as pkl
    with open(f'output/{RUN_NAME}/raw_suggested_formulations.pkl', 'wb') as f:
      pickle.dump(optimized_formulations, f)

    data_file_path = f'datasets/{DATASET_NAMES[cell_type_list[0]]}.xlsx' #ASSIGNED TO FIRST CELL IN CELL LIST
    df_existing = pd.read_excel(data_file_path)
    last_label = df_existing['Formula_label'].max() 
    last_iter = df_existing['Iter'].max()
    last_ionizable = df_existing['Ionizable_Lipid'].dropna().iloc[-1]
    last_helper = df_existing['Helper_lipid'].dropna().iloc[-1]

    cols = df_existing.columns.tolist()
    output_file_path = f'output/{RUN_NAME}/optimized_formulation_dataset.csv'

    print_slowly("\n\n--- EXPORTING ALL GENERATED FORMULATIONS to csv ---")

    # Create rows to append
    new_rows = []
    for i, row in optimized_formulations.iterrows():
        new_row = {
            'Formula_label': last_label + i + 1,
            'Iter': last_iter + 1,
            'Opt_Method': row['opt_method'],
            'Ionizable_Lipid': last_ionizable,
            'Helper_lipid': last_helper,
        }
        for param in input_param_names:
            new_row[param] = row[param]
        new_rows.append(new_row)

    # Create DataFrame from new rows
    df_new = pd.DataFrame(new_rows)

    # Create full structure if needed
    df_new_full = pd.DataFrame(columns=cols)
    df_new_full = pd.concat([df_new_full, df_new], ignore_index=True)

    # Combine with existing data and save
    df_combined = pd.concat([df_existing, df_new_full], ignore_index=True)
    df_combined.to_csv(output_file_path, index=False)
    print(f"Optimized formulations saved to {output_file_path}")


  if run_mantis_formatter == True:
    with open(f'output/{RUN_NAME}/raw_suggested_formulations.pkl', 'rb') as f:
      optimized_formulations = pickle.load(f)
    run_mantis_formatter_pipeline(RUN_NAME, input_param_names, optimized_formulations)

if __name__ == "__main__":
    # startup_banner()
    # Run the main function
    main()