
from falcon_engine.utilities import extract_training_data, init_pipeline, save_pipeline, startup_banner
from falcon_engine.run_Model_Selection import run_Model_Selection 
from falcon_engine.run_optimization import run_optimization_pipeline
from falcon_engine import learning_curve
import pandas as pd

"""
run_FALCON script

- main script to run FALCON computational pipeline 
- trains XGBoost surrogate models to predict LNP transfection in specified cell types
- leverages ML-guided search algorithms to suggest optimized LNP compositions for experimental testing. 
- Follow 4 steps to configure the pipeline, then run the script
- Find saved trained model and suggested formulations in output folder

"""

def main():

  ############### STEP 1: SEARCH CONFIGURATION #########################
  opt_method = 'BO' # DA or BO or NSGAII
  num_formulations = 5 #Default = 12 

  ############### STEP 2: CELL TYPES AND OBJECTIVE CONFIGURATION #######
  # ex. cell types used in manuscript ['RAMOS','DC','3T3','C2C12'] 
  MAX_cell_targets = ['RAMOS']
  MIN_cell_targets = [] # set as empty list if no minimization is desired (not '') 
  #MIN_cell_targets = ['DC','3T3','C2C12']

  #model training will be done for each cell type in this list
  #DA and BO will only use first cell type in this list for maximization, NSGAII will use all cell types
  cell_type_list = MAX_cell_targets + MIN_cell_targets 

  ################ STEP 3: LOAD AND SAVE PATH CONFIGURATION #############
  RUN_NAME = "0602_test" #Give a name for run folder to save any trained models
  DATASET_NAME = 'Normalized_RAMOS_THP1_Dual_Objective_4ITER_DSPC_DlinMC3DMA' #Name of the csv file, used to extract training data

  ################ STEP 4: PIPELINE COMPONENTS CONFIGURATION #############
  run_model_training = True # set true unless model is already trained and saved in output folder
  run_optimization = True # set true unless de novo formulation generation is not desired 
  run_mantis_formatter = False # set true if you want to format the optimized formulations for MANTIS (liquid handler) input

  ########################################################################
  data_file_path = f'datasets/{DATASET_NAME}.csv' #Path to the dataset to be used for training

  #Input_Params (do not change unless you change the dataset column order)
  input_param_names = ['NP_ratio',
                        'PEG_PEG+Chol',
                        'IL+HL',
                        'HL_IL+HL'] 

  if run_model_training == True:  
    LnRLU_floor = -1000 #cutoff below which LnRLU values are considered 0
    for c in cell_type_list:   #Loop through model training for each cell type of interest
      pipeline_path = f'output/{RUN_NAME}/{c}/Pipeline_dict.pkl'
      #Initialize new model pipeline
      pipeline_dict = init_pipeline(pipeline_path = pipeline_path,
                                      RUN_NAME=RUN_NAME,
                                      input_param_names=input_param_names,
                                      cell = c,
                                      model_list=['XGB'], #List of models to be trained, currently only XGB is supported for surrogate modeling
                                      param_type = 'percent',
                                      data_file_path=data_file_path,
                                      prefix='LnRLU_',
                                      RLU_floor=LnRLU_floor,
                                      N_CV=5)
      pipeline_dict, _, _, _= extract_training_data(pipeline_dict) 
      pipeline_dict, _, _, _ = run_Model_Selection(pipeline_dict)
      pipeline_dict = learning_curve.get_learning_curve(pipeline_dict, refined = False)
      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, step = 'FINAL SAVE')  
  
  if run_optimization == True:
    optimized_formulations = run_optimization_pipeline(opt_method, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME)
    # Save the optimized formulations to a file
    optimized_formulations = pd.DataFrame(optimized_formulations)
    df_existing = pd.read_csv(data_file_path)
    last_label = df_existing['Formula_label'].max() 
    last_iter = df_existing['Iter'].max()
    last_ionizable = df_existing['Ionizable_Lipid'].dropna().iloc[-1]
    last_helper = df_existing['Helper_lipid'].dropna().iloc[-1]

    cols = df_existing.columns.tolist()
    output_file_path = f'output/{RUN_NAME}/optimized_formulation_dataset.csv'

    # Create rows to append
    new_rows = []
    for i, (x, y_dict) in enumerate(zip(optimized_formulations[0], optimized_formulations[1])):
        new_row = {
            'Formula_label': last_label + i + 1,
            'Iter': last_iter + 1,
            'Opt_Method': opt_method,
            'Ionizable_Lipid': last_ionizable,
            'Helper_lipid': last_helper,
            'NP_ratio': x[0],
            'PEG_PEG+Chol': x[1],
            'IL+HL': x[2],
            'HL_IL+HL': x[3],
        }
        new_rows.append(new_row)
    df_new = pd.DataFrame(new_rows)
    df_new_full = pd.DataFrame(columns=cols)  # full structure
    df_new_full = pd.concat([df_new_full, df_new], ignore_index=True)

    # Combine and save
    df_combined = pd.concat([df_existing, df_new_full], ignore_index=True)
    df_combined.to_csv(output_file_path, index=False)
    print(f"Optimized formulations saved to {output_file_path}")

if __name__ == "__main__":
    startup_banner()
    # Run the main function
    main()