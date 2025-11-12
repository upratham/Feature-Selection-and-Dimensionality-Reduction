import pandas as pd
import logging
import os
from data_import import load_data


# Create Log directory
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#Create Logger
logger=logging.getLogger('data_preprocess')
logger.setLevel("DEBUG")

#Stream console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Set log file path
log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define Formatter
log_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

def data_preprocess( data=load_data()):
    try:
        numeric_cols=data.select_dtypes(include='number').columns # Apply to only numeric columns
        data[numeric_cols] = data[numeric_cols].mask(data[numeric_cols] < 0, 0) #This replaces all values in X that are less than 0 with 0.
        label_map = {'Good': 0, 'Moderate': 1, 'Poor': 2, 'Hazardous': 3}
        data['Air Quality'] = data['Air Quality'].map(label_map)
        logger.debug('Data Preprocessing completed')
        return data
    except Exception as e:
        logger.error('Unexpected error occured during data pre-processing %s',e)
        raise

def main():
    data=data_preprocess()
    print(data.head)

if __name__=='__main__':
    main()



