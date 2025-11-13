import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import data_import
import data_preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
import os
import feature_engineering

# Create Log directory

log_dir='logs'
# os.makedirs(log_dir,exist_ok=True)

#Create Logger
logger=logging.getLogger('model_training')
logger.setLevel("DEBUG")

#Stream console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Set log file path
log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define Formatter
log_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def train(data):
   
    X=data.drop(columns='Air Quality')
    y=data['Air Quality']
    # feature_names= np.array(X.columns.tolist())
    # feature_names
    # k=[1,2,3]
    # for i in k:
    #     df=feature_engineering.univeriate(i,X,y)
    #     df=feature_engineering.feature_imp_score(i,X,y)
    #     df=feature_engineering.pca(i,X,y)
    #     df=feature_engineering.lda(i,X,y)
    df=feature_engineering.lda(2,X,y)
    df=pd.DataFrame(df)
    print(df.head())



def main():
    data=data_import.load_data()
    data=data_preprocessing.data_preprocess(data)

    train(data)

if __name__ == '__main__':
    main()