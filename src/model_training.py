import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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




def train(data):
    results = {
    'model': []}
    
    K=[1,2,3]
    X=data.drop(columns='Air Quality')
    y=data['Air Quality']
    dim_reduction=[feature_engineering.lda,feature_engineering.pca,feature_engineering.univeriate,feature_engineering.feature_imp_score]
    dim_reduction_name=['LDA','PCA','Univeriate','Feat_imp_score']
    svm = SVC(kernel='linear')       # You can change kernel to 'rbf', 'poly', etc.
    guass_nb = GaussianNB()
    for funct,k in zip(dim_reduction ,K):
     
        df=funct(k,X,y)
        df=pd.DataFrame(df)
        for i in tqdm(range(2)):
            X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.25,random_state=None)
            # Linear SVC 
           
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            results['model'].append(['SVM',i,accuracy,dim_reduction_name[k],k])
            
            logger.debug('SVM trained, i=%s',{i},'accuracy=%s',{i},'dim_reduction:%s',{dim_reduction_name[k]},'K=%s',{k})
            
            #GaussianNB
            guass_nb.fit(X_train, y_train)
            gauss_preds = guass_nb.predict(X_test)
            accuracy=accuracy_score(y_test,gauss_preds)
            results['model'].append(['guass_nb',i,accuracy,dim_reduction_name[k],k])
            logger.debug('guass_nb trained, i=%s',{i},'accuracy=%s',{i},'dim_reduction:%s',{dim_reduction_name[k]},'K=%s',{k})
            
    
    result_df = pd.DataFrame(results['model'], columns=['Model', 'Fold', 'Accuracy', 'Method', 'K'])
    save_dir = "result/"
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "model_results.csv"), index=False)
     
    return result_df
        


def main():
    data=data_import.load_data()
    data=data_preprocessing.data_preprocess(data)

    result=train(data)
    result.shape()

if __name__ == '__main__':
    main()