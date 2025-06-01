
from falcon_engine.utilities import extract_training_data, init_pipeline, save_pipeline, startup_banner
from falcon_engine.run_Model_Selection import run_Model_Selection 
from falcon_engine import learning_curve
import pickle

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
  opt_method = 'DA' # DA or BO or NSGAII
  num_formulations = 12 #Default = 12 

  ############### STEP 2: CELL TYPES AND OBJECTIVE CONFIGURATION #######
  # ex. cell types used in manuscript ['RAMOS','DC','3T3','C2C12'] 
  MAX_cell_targets = ['RAMOS']
  MIN_cell_targets = []
  #MIN_cell_targets = ['DC','3T3','C2C12']

  #model training will be done for each cell type in this list
  #DA and BO will only use first cell type in this list, NSGAII will use all cell types
  cell_type_list = MAX_cell_targets + MIN_cell_targets 

  ################ STEP 3: LOAD AND SAVE PATH CONFIGURATION #############
  RUN_NAME = "0530_test" #Give a name for run folder to save any trained models
  DATASET_NAME = 'Normalized_RAMOS_Single_Objective_3ITER_DSPC_DlinMC3CMA' #Name of the csv file, used to extract training data

  ################ STEP 4: PIPELINE COMPONENTS CONFIGURATION #############
  run_model_training = True # set true unless model is already trained and saved in output folder
  run_optimization = True # set true unless de novo formulation generation is not desired 

  ########################################################################
  data_file_path = f'datasets/{DATASET_NAME}.csv' #Path to the dataset to be used for training

  if run_model_training == True:  
    LnRLU_floor = 2.5 #cutoff below which LnRLU values are considered 0
    for c in cell_type_list:   #Loop through model training for each cell type of interest
      pipeline_path = f'output/{RUN_NAME}/{c}/Pipeline_dict.pkl'
      #Initialize new model pipeline
      pipeline_dict = init_pipeline(pipeline_path = pipeline_path,
                                      RUN_NAME=RUN_NAME,
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
    #Run optimization for each cell type
    for c in MAX_cell_targets:
      pipeline_path = f'output/{RUN_NAME}/{c}/Pipeline_dict.pkl'
      with open(pipeline_path, 'rb') as file:
        pipeline_dict = pickle.load(file)
      #Run optimization
      from falcon_engine.run_optimization import run_optimization
      run_optimization(pipeline_dict, opt_method=opt_method, num_formulations=num_formulations)

if __name__ == "__main__":
    startup_banner()
    # Run the main function
    main()