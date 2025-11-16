import data_import
import data_preprocessing
import model_training
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from feature_engineering import log_insight

# Create Log directory

log_dir='logs'
# os.makedirs(log_dir,exist_ok=True)

#Create Logger
logger=logging.getLogger('model_eval')
logger.setLevel("DEBUG")

#Stream console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Set log file path
log_file_path=os.path.join(log_dir,'model_eval.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define Formatter
log_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

results_file_path=os.path.join(r'result/result_eval_report.txt')
open(results_file_path, 'w').close()
plots_dir=os.path.join(r'result\Result_eval_plots')
os.makedirs(plots_dir,exist_ok=True)

def log_insight(msg: str):
    """Log insight to logger + write to results text file."""
    logger.info(msg)
    with open(results_file_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def call_train():
    data=data_import.load_data()
    data=data_preprocessing.data_preprocess(data)
    iter=50 # Different number of train test splits
    K=[1,2,3,9] # number of features all features =9
    result=model_training.train(data,iter=iter,K=K)
    print(result.head)
    print(result.describe(include='all'))

def eval():
    result=pd.read_csv(r'result\model_training_results.csv')
    result.head()
    
    print(result.describe(include='all'))
    msg='Result.describe all'
    log_insight(msg)
    msg=str(result.describe(include='all'))
    log_insight(msg)

    dim_reduction_name=['Univeriate','Feat_imp_score','PCA','LDA']

    for i in dim_reduction_name:
        df = result.copy()
        df = df[df['Method'] == i].copy()

        df['K'] = pd.to_numeric(df['K'])
        
        mean_acc = ( df.groupby(['K', 'Model'], as_index=False)['Accuracy'].mean())
        mean_acc_per_k = (df.groupby('K', as_index=False)['Accuracy'].mean())
        msg=f"Average accuracy per K (all models combined) for metho ={i}: {mean_acc_per_k}"
        log_insight(msg)
        
        
        msg=f"\nAverage accuracy per K per Model for method= {i}\n {mean_acc.pivot(index='K', columns='Model', values='Accuracy')}"
        log_insight(msg)
        
        plt.figure()

        for model, grp in mean_acc.groupby('Model'):
            grp = grp.sort_values('K')          # ensure K is in order 1,2,3
            plt.plot(grp['K'], grp['Accuracy'],
                    marker='o',                 # dots on points
                    label=model)                # legend entry

        plt.xlabel('K')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Accuracy vs K by Model (Method = {i})')
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(plots_dir, f'Accuracy vs K by Model (Method = {i}).png')
        plt.savefig(fig_path)
        plt.show()

def main():
    call_train()
    eval()

if __name__ == '__main__':
    main()
