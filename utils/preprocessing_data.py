import numpy as np
import pandas as pd
import os
import warnings
import random
import math

from data import ActivityDataset

activity_codes_mapping = {'A': 'walking',
                          'B': 'jogging',
                          'C': 'stairs',
                          'D': 'sitting',
                          'E': 'standing',
                          'F': 'typing',
                          'G': 'brushing teeth',
                          'H': 'eating soup',
                          'I': 'eating chips',
                          'J': 'eating pasta',
                          'K': 'drinking from cup',
                          'L': 'eating sandwich',
                          'M': 'kicking soccer ball',
                          'O': 'playing catch tennis ball',
                          'P': 'dribbling basket ball',
                          'Q': 'writing',
                          'R': 'clapping',
                          'S': 'folding clothes'}

leg_activities = {'A': 'walking',
                  'B': 'jogging',
                  'C': 'stairs',
                  'D': 'sitting',
                  'E': 'standing',
                  'M': 'kicking soccer ball',
                  'O': 'playing catch tennis ball',
                  'P': 'dribbling basket ball',
                 }

legcodes_mapping ={'A': 0,
                  'B': 1,
                  'C': 2,
                  'D': 3,
                  'E': 4,
                  'M': 5,
                  'O': 6,
                  'P': 7,
                  }

# Resampling the data
# np.searchsorted find the indices where t_new are inserted
# np.maximum element wise compare the two array
# Have to resampling the data because of non-uniform sampling time intervals
def resample(t_0, x_0, t_new, epsilon=1e-6):
    """ Resample value at t_new using linear interpolation
        Args:
            t_0: times of the raw data (list of sampling times)
            x_0: information corresponding to each of t_0
            t_new: set of new sampling times which have a consistent sampling interval
        Returns:
            x_new: list of new information corresponding to t_new
    """
    ind = np.searchsorted(t_0, t_new, side='right')
    
    lower_ind = np.maximum(0, ind-1)
    upper_ind = np.minimum(len(t_0)-1, ind)
    
    lower_t = t_0[lower_ind]
    upper_t = t_0[upper_ind]
    
    lower_x = x_0[lower_ind]
    upper_x = x_0[upper_ind]
    
    denom = upper_t - lower_t
    denom_mask = denom < epsilon
    x_new = lower_x + (upper_x - lower_x) * (t_new - lower_t) / denom
    x_new[denom_mask] = lower_x[denom_mask]
    
    return x_new

def resampling_data(dt):
    """
        args:
            dt: resampling time step
        returns:
            list of dictionaries of participants with activities as keys
    """
    dt  = 50 #time step

    resampled_pars = []
    warnings.filterwarnings(action = 'ignore')
    par_id  = [i for i in range(51)] #include 51 participants 
    for id in par_id:
        activity_data = ActivityDataset(id)
        raw_pars_accel = activity_data._load_raw_data('accel', 'phone', '../assets/wisdm-dataset/raw', 'a')
        raw_pars_gyro = activity_data._load_raw_data('gyro', 'phone', '../assets/wisdm-dataset/raw', 'g')
    
        activities = {}
        for k in leg_activities:
            resample_vals = {}
            
            raw_pars_accel_act_mask = raw_pars_accel['activity_code'] == k
            raw_pars_gyro_act_mask = raw_pars_gyro['activity_code'] == k
            
            act_data_accel = raw_pars_accel[raw_pars_accel_act_mask]
            act_data_gyro = raw_pars_gyro[raw_pars_gyro_act_mask]
            
            #if there is no file for participant with the id
            if act_data_accel.empty or act_data_gyro.empty:
                continue
            
            act_data_t_min = max(act_data_accel['timestamp'].min(), act_data_gyro['timestamp'].min())
        
            act_data_accel['timestamp_shift_ms'] = (act_data_accel['timestamp'] - act_data_t_min)//int(1e6)
            act_data_gyro['timestamp_shift_ms'] = (act_data_gyro['timestamp'] - act_data_t_min)//int(1e6)
            
            act_data_s_t_max = min(act_data_accel['timestamp_shift_ms'].max(), act_data_gyro['timestamp_shift_ms'].max())
            t_sample = np.arange(0, act_data_s_t_max, dt) # Include participant data from time 0 to act_data_s_t_max
                
            resample_vals['t'] = t_sample
            
            resample_vals['ax0'] = resample(act_data_accel['timestamp_shift_ms'].values, act_data_accel['ax'].values, t_sample)
            resample_vals['ay0'] = resample(act_data_accel['timestamp_shift_ms'].values, act_data_accel['ay'].values, t_sample)
            resample_vals['az0'] = resample(act_data_accel['timestamp_shift_ms'].values, act_data_accel['az'].values, t_sample)
            
            resample_vals['gx0'] = resample(act_data_gyro['timestamp_shift_ms'].values, act_data_gyro['gx'].values, t_sample)
            resample_vals['gy0'] = resample(act_data_gyro['timestamp_shift_ms'].values, act_data_gyro['gy'].values, t_sample)
            resample_vals['gz0'] = resample(act_data_gyro['timestamp_shift_ms'].values, act_data_gyro['gz'].values, t_sample)
            
            activities[k] = pd.DataFrame(resample_vals)
            
        resampled_pars.append(activities)
    
    return resampled_pars

def stack_columns(resampled_pars, par_id: list, num_stack: int):
    """
        stacking pandas columns of accerelation and gyration at different time steps

        Args: 
            resampled_pars: resampled dataset for all participants
            id: id of considered participants
            num_stack: number of time step in one sliding window (considered as one dataset)

        Returns:
            pars_stacked: new pandas dataframe with stacked columns
    """
    #Preparing the data sets
    warnings.filterwarnings(action = 'ignore')

    # Time window with num_stack time steps
    # From the data each activities contain around 3200 time steps, so we extract the data for 3200 time steps which give floor(3200/num_stack) data per activities.
    pars_stacked = pd.DataFrame({})
    for i in par_id:
        for k in resampled_pars[i].keys():
            for chunk in range(math.floor(3200/num_stack)): 
                
                raw_pars_train = resampled_pars[i][k][chunk*num_stack:(chunk+1)*num_stack].copy()
                for m in range(1, num_stack):
                    raw_pars_train['ax'+str(m)] = raw_pars_train['ax'+str(0)].shift(periods = -m)
                    raw_pars_train['ay'+str(m)] = raw_pars_train['ay'+str(0)].shift(periods = -m)
                    raw_pars_train['az'+str(m)] = raw_pars_train['az'+str(0)].shift(periods = -m)
                    raw_pars_train['gx'+str(m)] = raw_pars_train['gx'+str(0)].shift(periods = -m)
                    raw_pars_train['gy'+str(m)] = raw_pars_train['gy'+str(0)].shift(periods = -m)
                    raw_pars_train['gz'+str(m)] = raw_pars_train['gz'+str(0)].shift(periods = -m)
                raw_pars_train.dropna(inplace = True)
                raw_pars_train.drop('t', axis='columns', inplace=True)
                raw_pars_train['labels'] = k

                pars_stacked = pd.concat([pars_stacked, raw_pars_train], ignore_index=True)

    return pars_stacked

def save_processed_data(pars_stacked, data_type):

    if data_type.lower() not in ["train", "test", "validation"]:
        raise ValueError("Invalid data type. Please provide 'train', 'test', or 'validation'.")
    
    #Shuffling the index
    pre_shuffle = pars_stacked.copy()
    post_shuffle = pre_shuffle.sample(frac = 1).reset_index()
    post_shuffle.drop('index', axis='columns', inplace=True)


    labels = post_shuffle['labels']   
    for k in leg_activities:
        labels.replace(k, legcodes_mapping[k], inplace =True)

    features = post_shuffle.copy()
    features.drop('labels', axis='columns', inplace=True)

    #The post-processing data is downloaded into csv file. After compiling this, we don't need to recompile this again. 
    #Just read the downloaded csv file before training the process.

    # Check if the file exists and delete it if it does
    features_file_name = "../assets/postprocessing_dataset/" + "processed_pars_" + data_type + "_features.csv"
    if os.path.exists(features_file_name):
        os.remove(features_file_name)
        print(f"Removed outdated version of {features_file_name}")

    labels_file_name = "../assets/postprocessing_dataset/" + "processed_pars_" + data_type + "_labels.csv"
    if os.path.exists(features_file_name):
        os.remove(features_file_name)
        print(f"Removed outdated version of {features_file_name}")


    features.to_csv(features_file_name, index = False)
    labels.to_csv(labels_file_name, index = False)

def main():
    """
        Using non-overlapping sliding window method to create trainig/validating/testing dataset
    """
    # Randomly choosing 72% (36 participants) of 51 participant as train set, 8% (4 participants) of 51 participants as validation set and the remaining 20% (11 participants) as test set.
    ID = [num for num in range(51)]
    random.shuffle(ID)
    train_id = ID[:36]
    validation_id = ID[36:40]
    test_id = ID[40:51]

    # resampling all data (clean data)
    dt = 50 #time step of each sampling point is 50
    resampled_pars = resampling_data(dt)

    pars_train_stacked = stack_columns(resampled_pars, train_id, 64)
    save_processed_data(pars_train_stacked, "train")

    pars_test_stacked = stack_columns(resampled_pars, test_id, 64)
    save_processed_data(pars_test_stacked, "test")

    pars_validation_stacked = stack_columns(resampled_pars, validation_id, 64)
    save_processed_data(pars_validation_stacked, "validation")

if __name__ == "__main__":
    main()



 

