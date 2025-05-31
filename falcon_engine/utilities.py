import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def init_pipeline(pipeline_path, RUN_NAME, cell, param_type, data_file_path, prefix, RLU_floor, N_CV, model_list ):
    
    print('\n\n########## INITIALIZING MODEL TRAINING PIPELINE ##############\n\n')
    #Saving/Loading
    model_save_path           = f"output/{RUN_NAME}/{cell}/" # Where to save model, results, and training data 
    
    #Input_Params
    input_param_names = ['NP_ratio',
                        'PEG_PEG+Chol',
                        'IL+HL',
                        'HL_IL+HL'] 
    print(f"INPUT PARAMS: {input_param_names}")
    
    #initialize Pipeline Config and Data Storage Dictionary
    pipeline_dict = {'Cell' : cell,
                    'STEPS_COMPLETED':{
                        'Preprocessing': False,
                        'Model_Selection': False,
                        'Learning_Curve': False
                        },
                    'Saving':{
                        'RUN_NAME': RUN_NAME,
                        'Models': model_save_path,
                        },
                    'Data_preprocessing': {
                        'Data_Path': data_file_path,
                        'Formula_param_type': param_type,
                        'Input_Params': input_param_names,
                        'prefix' : prefix,
                        'RLU_floor':RLU_floor, 
                        'Scaler': None,
                        'X' : None, 
                        'y': None, 
                        'all_proc_data' : None,
                        'raw_data' : None
                        },
                    'Model_Selection': {
                        'Method': 'Nested CV',
                        'N_CV' : N_CV,
                        'Model_list': model_list,
                        'NESTED_CV': {},
                        'Best_Model':{
                            'Model_Name' : None, 
                            'Model': None, 
                            'Hyper_Params': None,
                            'Predictions' : None,
                            'MAE': None,
                            'Spearman':None,
                            'Pearson': None,
                            'HL_1': None
                            },
                            
                        },
                    'Learning_Curve':
                    {'NUM_ITER': None,
                        'num_splits': None,
                        'num_sizes': None,
                        'Train_Error': None,
                        'Valid_Error': None
                        }
                    }
  
    ####check save paths ########

    if not os.path.exists(model_save_path):
        # Create the directory if it doesn't exist
        os.makedirs(model_save_path)
        print(f"Directory '{model_save_path}' created.")
    else:
        print(f"Directory '{model_save_path}' already exists.")

    with open(pipeline_path , 'wb') as file:
        pickle.dump(pipeline_dict, file)
        print(f"\n\n--- SAVED New {cell} Pipeline CONFIG  ---")
    return pipeline_dict    
def save_pipeline(pipeline, path, step):
    c = pipeline['Cell']
    with open(path , 'wb') as file:
            pickle.dump(pipeline, file)
    print(f"\n--- SAVED PIPELINE: {step} CONFIG AND RESULTS for {c}  ---")
def extract_training_data(pipeline):
    #Assign variables based on dictionary
    cell_type = pipeline['Cell']
    data_path = pipeline['Data_preprocessing']['Data_Path']
    input_params = pipeline['Data_preprocessing']['Input_Params']
    prefix = pipeline['Data_preprocessing']['prefix']
    RLU_floor = pipeline['Data_preprocessing']['RLU_floor']
    
    #Extract datafile
    df = pd.read_csv(data_path)

    #Formatting Training Data
    raw_data = df[['Formula_label', 'Helper_lipid'] + input_params + [prefix + cell_type]].copy()
    raw_data = raw_data.dropna() #Remove any NaN rows

    processed_data = raw_data.copy()

    #floor all RLU values below the noise
    processed_data.loc[processed_data[prefix + cell_type] < RLU_floor, prefix + cell_type] = RLU_floor 

    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(processed_data.index))

    X = processed_data[input_params]                         
    Y = processed_data[prefix + cell_type].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = ["Scaled_" + prefix + cell_type])

    #Update Pipeline dictionary
    pipeline['Data_preprocessing']['Scaler'] = scaler
    pipeline['Data_preprocessing']['X'] = X
    pipeline['Data_preprocessing']['y'] = Y
    pipeline['Data_preprocessing']['all_proc_data'] = processed_data
    pipeline['Data_preprocessing']['raw_data'] = raw_data
    pipeline['STEPS_COMPLETED']['Preprocessing'] = True

    return pipeline, X,Y, processed_data