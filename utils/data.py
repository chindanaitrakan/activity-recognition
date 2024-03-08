import numpy as np
import pandas as pd

class ActivityDataset():

    def __init__(self, person_id):
        self.person_id = person_id

    def _load_raw_data(self, sensor, device, base_path, col_prefix):
        """
        Args:
            sensor: accel or gyro
            device: phone or watch
            base_path: base path of the data files
            col_prefix: a (accelerometer) or g (gyroscope)

        Returns:
            raw_par: information from a user i in pandas object 
        """
        
        id = self.person_id
        file_path = f'{base_path}/{device}/{sensor}/data_16{id:02}_{sensor}_{device}.csv'
        raw_par = pd.read_csv(file_path, names = ['participant_id' , 'activity_code' , 'timestamp', 
                                                f'{col_prefix}x', f'{col_prefix}y', f'{col_prefix}z'], 
                            index_col=None, header=None)

        return raw_par


