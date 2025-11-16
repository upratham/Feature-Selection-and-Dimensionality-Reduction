import pandas as pd
import data_import
import data_preprocessing
import logging
import os
import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tqdm.auto  import tqdm
from sklearn.metrics import accuracy_score
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




def train(data,iter,K):
    msg="=== NEW train() CALL ==="
    logger.debug(msg)
    results = []
    X=data.drop(columns='Air Quality')
    y=data['Air Quality']
    dim_reduction=[feature_engineering.univeriate,feature_engineering.feature_imp_score,feature_engineering.pca,feature_engineering.lda]
    dim_reduction_name=['Univeriate','Feat_imp_score','PCA','LDA']
    svm = SVC(kernel='linear')     
    guass_nb = GaussianNB()
    for funct in dim_reduction:
        idx = dim_reduction.index(funct)
        print(idx)
        for k in K:
            if k>3:
                df=X
            else:
                df=funct(k,X,y)             
            df=pd.DataFrame(df)
            for i in range(iter):
                
                X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.25,random_state=None)

               
                # Linear SVC 
            
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                accuracy=accuracy_score(y_test,y_pred)
                results.append([i,'SVM',accuracy,dim_reduction_name[idx],k])
                msg=f'i= {i} SVM trained, accuracy={accuracy} dim_reduction:{dim_reduction_name[idx]} K={k}'
                logger.debug(msg)
                
                #GaussianNB
                guass_nb.fit(X_train, y_train)
                gauss_preds = guass_nb.predict(X_test)
                accuracy=accuracy_score(y_test,gauss_preds)
                results.append([i,'guass_nb',accuracy,dim_reduction_name[idx],k])
                msg=f'i= {i},guass_nb trained, accuracy={accuracy} dim_reduction:{dim_reduction_name[idx]} K={k}'
                logger.debug(msg)
            
    
    result_df = pd.DataFrame(results,columns=[ 'Fold','Model', 'Accuracy', 'Method', 'K'])
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "model_training_results.csv"), index=False)
        
    return result_df
        


def main():
    data=data_import.load_data()
    data=data_preprocessing.data_preprocess(data)

    result=train(data,5)
    result.shape

if __name__ == '__main__':
    main()