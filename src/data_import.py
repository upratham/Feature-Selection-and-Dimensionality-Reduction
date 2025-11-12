import pandas as pd
import logging
import os

# Create Log directory
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#Create Logger
logger=logging.getLogger('data_import')
logger.setLevel("DEBUG")

#Stream console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Set log file path
log_file_path=os.path.join(log_dir,'data_import.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define Formatter
log_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

#Complete logger

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def load_data(path='data\pollution_dataset.csv')->pd.DataFrame:
    try:
        data=pd.read_csv(path)
        logger.debug('Data loaded from %s',path)
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the file: %s',e)
        raise


def main():
    data=load_data()
    print(data.head())

if __name__ == '__main__':
    main()