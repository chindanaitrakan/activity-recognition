import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

def show_accel_per_activity(device, df, act, participant_id, interval_in_sec = None):
    ''' Plots acceleration time history per activity '''

    df1 = df.loc[df.activity_code == act].copy()
    df1.reset_index(drop = True, inplace = True)

    act = activity_codes_mapping[act]

    df1['duration'] = (df1['timestamp'] - df1['timestamp'].iloc[0])/1000000000 # nanoseconds --> seconds

    if interval_in_sec == None:
        ax = df1[:].plot(kind='line', x='duration', y=['ax','ay','az'], figsize=(25,7), grid = True)
    else:
        ax = df1[:interval_in_sec*20].plot(kind='line', x='duration', y=['ax','ay','az'], figsize=(25,7), grid = True)

    ax.set_xlabel('duration  (sec)', fontsize = 15)
    ax.set_ylabel('acceleration  (m/sec^2)',fontsize = 15)
    ax.set_title('Acceleration:   Device: ' + device + '     Participant ID: ' + '16'+f'{participant_id:02}' + '      Activity:  ' + act, fontsize = 15)

    # Check if the file exists and delete it if it does
    file_name = "../assets/visualization/time_series/accel_participant"+"16"+f'{participant_id:02}'+"_"+f'{act}'+"_timeseries.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)

def show_ang_velocity_per_activity(device, df, act, participant_id, interval_in_sec = None):
    ''' Plots angular volocity time history per activity '''

    df1 = df.loc[df.activity_code == act].copy()
    df1.reset_index(drop = True, inplace = True)

    act = activity_codes_mapping[act]

    df1['duration'] = (df1['timestamp'] - df1['timestamp'].iloc[0])/1000000000 # nanoseconds --> seconds

    if interval_in_sec == None:
        ax = df1[:].plot(kind='line', x='duration', y=['gx','gy','gz'], figsize=(25,7), grid = True) # ,title = act)
    else:
        ax = df1[:interval_in_sec*20].plot(kind='line', x='duration', y=['gx','gy','gz'], figsize=(25,7), grid = True) # ,title = act)

    ax.set_xlabel('duration  (sec)', fontsize = 15)
    ax.set_ylabel('angular velocity  (rad/sec)',fontsize = 15)
    ax.set_title('Angular velocity:  Device: ' + device + '     Participant ID: ' + '16'+f'{participant_id:02}' + '      Activity:  ' + act, fontsize = 15)

    # Check if the file exists and delete it if it does
    file_name = "../assets/visualization/time_series/gyro_participant"+"16"+f'{participant_id:02}'+"_"+f'{act}'+"_timeseries.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)

def show_acceleration_statistics(df, act, participant_id):
    #Draw Histogram and Kde for each activities within the time
    #Accelerometer
    if df[df['activity_code'] == act].empty:
        return
    T = df[df['activity_code'] == act]

    f, axes = plt.subplots(1, 2, figsize=(16,8))
    a = sb.histplot(T[['ax', 'ay', 'az']], kde = True, bins=100, ax=axes[1])
    ID = int(T['participant_id'][:1])
    Activity = activity_codes_mapping[act]
    a.set(xlabel ="Acceleration m/s^2", ylabel = "Count", title = 'Participant_'+f'{ID}'+'  Activity: '+f'{Activity}')
    a_box = sb.boxplot(data=T[['ax', 'ay', 'az']], orient = "h", ax = axes[0])
    a_box.set(xlabel ="Acceleration m/s^2", ylabel = "Count", title = 'Participant_'+f'{ID}'+'  Activity: '+f'{Activity}')

    #Check if the file exists and delete it if it does
    file_name = "../assets/visualization/individual_statistics/accel_participant"+"16"+f'{participant_id:02}'+"_"+f'{activity_codes_mapping[act]}'+"_statistics.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)
    return

    

def show_gyration_statistics(df, act, participant_id):
    #Draw Histogram and Kde for each activities within the time
    #Gyroscope
    if df[df['activity_code'] == act].empty:
        return
    T = df[df['activity_code'] == act]
        
    f, axes = plt.subplots(1, 2, figsize=(16,8))
    g = sb.histplot(T[['gx', 'gy', 'gz']], kde = True, bins=100)
    ID = int(T['participant_id'][:1])
    Activity = activity_codes_mapping[act]
    g.set(xlabel ="Angular velocity rad/s", ylabel = "Count", title = 'Participant_'+f'{ID}'+'  Activity: '+f'{Activity}')
    g_box = sb.boxplot(data=T[['gx', 'gy', 'gz']], orient = "h", ax = axes[0])
    g_box.set(xlabel ="Angular velocity rad/s", ylabel = "Count", title = 'Participant_'+f'{ID}'+'  Activity: '+f'{Activity}')

    # Check if the file exists and delete it if it does
    file_name = "../assets/visualization/individual_statistics/gyro_participant"+"16"+f'{participant_id:02}'+"_"+f'{activity_codes_mapping[act]}'+"_statistics.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)

def main():

    #load raw data of all participant
    par_id  = [i for i in range(51)] #include 51 participants 
    raw_data_accel = []
    raw_data_gyro = []
    for id in par_id:
        activity_data = ActivityDataset(id)
        raw_pars_accel = activity_data._load_raw_data('accel', 'phone', '../assets/wisdm-dataset/raw', 'a')
        raw_pars_gyro = activity_data._load_raw_data('gyro', 'phone', '../assets/wisdm-dataset/raw', 'g')
        raw_data_accel.append(raw_pars_accel)
        raw_data_gyro.append(raw_pars_gyro)
    
    #save time series graph of participant_id = 0
    for key in activity_codes_mapping:
        for id in [0]:
            show_accel_per_activity('Phone', raw_data_accel[id], key, id, 20)
            show_ang_velocity_per_activity('Phone', raw_data_gyro[id], key, id, 20)

    #save individual statistical representation of each activity of each participant
    par_id = [0] #chosen participant
    for key in activity_codes_mapping:
        for id in par_id:
            show_acceleration_statistics(raw_data_accel[id], key, id)
            show_gyration_statistics(raw_data_gyro[id], key, id)

    #To observe the mean and variance, bloxplot is a good visualization. #looking at participation 1600
    raw_data_accel[0]['activity'] = raw_data_accel[0]['activity_code'].copy().map(activity_codes_mapping)
    f, axes = plt.subplots(1, 1, figsize=(16, 8))
    sb.boxplot(y = 'activity', x = 'ay', data = raw_data_accel[0])

    file_name = "../assets/visualization/statistics/accel_participant"+"16"+f'{0:02}'+"_statistics.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)

    #Looking at the same activity but different participant (compare among participants)

    raw_pars_accel = pd.DataFrame({})
    k = 'A' #Walking
    for i in range(51):
        raw_pars_accel = pd.concat([raw_pars_accel, raw_data_accel[i][raw_data_accel[i]['activity_code']==k]])
    raw_pars_accel_copy = {}
    raw_pars_accel_copy[k] = raw_pars_accel.copy()
    raw_pars_accel_copy[k]['participant_id'] = raw_pars_accel_copy[k]['participant_id'].astype('category')
    raw_pars_accel_copy[k]
    raw_pars_accel_copy[k].dtypes
    f, axes = plt.subplots(1, 1, figsize=(16, 8))

    sb.boxplot(y = 'participant_id', x = 'ax', data = raw_pars_accel_copy[k])
    file_name = "../assets/visualization/statistics/accelx_all_participants_" + f'{activity_codes_mapping["A"]}' + "_statistics.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")
    plt.savefig(file_name)

    f, axes = plt.subplots(1, 1, figsize=(16, 8))
    sb.boxplot(y = 'participant_id', x = 'ay', data = raw_pars_accel_copy[k])
    file_name = "../assets/visualization/statistics/accely_all_participants_" + f'{activity_codes_mapping["A"]}' + "_statistics.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")
    plt.savefig(file_name)

    f, axes = plt.subplots(1, 1, figsize=(16, 8))
    sb.boxplot(y = 'participant_id', x = 'az', data = raw_pars_accel_copy[k])
    file_name = "../assets/visualization/statistics/accelz_all_participants_" + f'{activity_codes_mapping["A"]}' + "_statistics.png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")
    plt.savefig(file_name)

if __name__ == "__main__":
    main()
