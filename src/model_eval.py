import data_import
import data_preprocessing
import model_training

data=data_import.load_data()
data=data_preprocessing.data_preprocess(data)
iter=50 # Different number of train test splits
K=[1,2,3] # number of features
result=model_training.train(data,iter=iter,K=K)
print(result.head)
print(result.describe(include='all'))
    
