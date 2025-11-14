import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import data_import
import data_preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import logging
import os


# Create Log directory

log_dir='logs'
# os.makedirs(log_dir,exist_ok=True)

#Create Logger
logger=logging.getLogger('feature_engineering')
logger.setLevel("DEBUG")

#Stream console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Set log file path
log_file_path=os.path.join(log_dir,'fearure_engineering.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define Formatter
log_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#crete result directory to save all results
result_dir='result'
os.makedirs(result_dir,exist_ok=True)
results_file_path=os.path.join(result_dir,'feature_engineering_result.txt')
open(results_file_path, 'w').close()
#craete plot directory to save the plots
plots_dir=os.path.join(result_dir,'feature_engineering_plots')
os.makedirs(plots_dir,exist_ok=True)

def log_insight(msg: str):
    """Log insight to logger + write to results text file."""
    logger.info(msg)
    with open(results_file_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def univeriate(i,X,y):
    number_features=i
    #UNiveriate feature selection
    feature_names= np.array(X.columns.tolist()) 
    selector=SelectKBest(chi2,k=number_features)   
    X_new = selector.fit_transform(X,y)
    X_new_features_mask = selector.get_support()
    X_new_feature_names = feature_names[X_new_features_mask]
    msg = f'\nSelected {number_features} features using univariate feature selection:\n{X_new_feature_names}'
    log_insight(msg)
    return X_new


def feature_imp_score(i,X,y):
    # Feature selection by importance score
    feature_names= np.array(X.columns.tolist())
    classifier=RandomForestClassifier(n_estimators=100,random_state=21)
    classifier.fit(X,y)
    feature_imp_scores=pd.DataFrame({'fea_imp_score': classifier.feature_importances_})
    feature_imp_scores['Features']=feature_names
    feature_imp_scores_sorted = feature_imp_scores.sort_values(by=feature_imp_scores.columns[0], ascending=False)
    top_k = feature_imp_scores_sorted['Features'].head(i).tolist()         # top k rows (names + scores)
    msg = f'\nSelected : {i} features using feature importance score:\n{top_k}'
    log_insight(msg)
    #Visualise the feature importance score
    plt.figure(figsize=(8,5))
    plt.bar(feature_imp_scores_sorted["Features"], feature_imp_scores_sorted["fea_imp_score"])
    plt.xlabel("Features")
    plt.ylabel("Feature Importance Score")
    plt.title(f"Feature Importance Plot k={i}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, f'feature_importance_top_{i}.png')
    plt.savefig(fig_path)
 

    model=SelectFromModel(classifier,prefit=True)
    X_new=model.transform(X)
    X_new_k=X_new[:i,:]
    return X_new_k
    


def pca(k,X,y):
    nComps=k
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=nComps)
    pca.fit(X_scaled)
    X_pca_k=pca.transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    plt.figure(figsize=(6, 5)) 
    classes = np.unique(y)
    colors = ['navy', 'turquoise', 'darkorange', 'red']

    if k == 1:
        # 1D case: PC1 vs class (stacked by class on y-axis)
        for cls, color in zip(classes, colors):
            idx = (y == cls)
            plt.scatter(
                X_pca_k[idx, 0],
                np.zeros(np.sum(idx)) + cls,
                label=f'class_{cls}',
                edgecolors='k',
                color=color
            )
        plt.xlabel('PC1')
        plt.ylabel('Class')
        plt.title('PCA (k=1): PC1 vs class')
        plt.legend()
        plt.tight_layout()
        scatter_path = os.path.join(plots_dir, f'pca_k_{k}_scatter.png')
        plt.savefig(scatter_path)
        
        

    elif k==2:
        # 2D case: PC1 vs PC2 colored by class (like your screenshot)
        for cls, color in zip(classes, colors):
            idx = (y == cls)
            plt.scatter(
                X_pca_k[idx, 0],
                X_pca_k[idx, 1],
                label=f'class_{cls}',
                edgecolors='k',
                color=color
            )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PCA (k={k}): PC1 vs PC2')
        plt.legend()
        plt.tight_layout()
        scatter_path = os.path.join(plots_dir, f'pca_k_{k}_scatter.png')
        plt.savefig(scatter_path)
      
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        for cls, color in zip(classes, colors):
            idx = (y == cls)
            ax.scatter(
                X_pca_k[idx, 0],
                X_pca_k[idx, 1],
                X_pca_k[idx, 2],
                label=f'class_{cls}',
                edgecolors='k',
                color=color
            )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA (k=3): PC1 vs PC2 vs PC3')
        ax.legend()
        plt.tight_layout()
        scatter_path = os.path.join(plots_dir, f'pca_k_{k}_scatter_3d.png')
        plt.savefig(scatter_path)
       



    # variance ration visualization 
    plt.figure(figsize=(5, 4))
    components = np.arange(1, len(var_ratio) + 1)
    plt.bar(components, var_ratio)
    plt.xticks(components)
    plt.xlabel('PCA')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA Explained Variance Ratio (k={k})')
    plt.tight_layout()

    var_path = os.path.join(plots_dir, f'PCA_k_{k}_var_ratio.png')
    plt.savefig(var_path)

    log_insight(f'PCA (k={k}) explained variance ratio: {var_ratio}')

    

    return X_pca_k
 


def lda(k,X,y):
    nComps = k
    X_scaled = StandardScaler().fit_transform(X)
    lda = LDA(n_components=nComps)
    lda.fit(X_scaled,y)
    X_lda_k = lda.transform(X_scaled)
    var_ratio = lda.explained_variance_ratio_

    plt.figure(figsize=(6, 5))
    classes = np.unique(y)
    colors = ['navy', 'turquoise', 'darkorange', 'red']

    if k == 1:
        for cls, color in zip(classes, colors):
            idx = (y == cls)
            plt.scatter(
                X_lda_k[idx, 0],
                np.zeros(np.sum(idx)) + cls,
                label=f'class_{cls}',
                edgecolors='k',
                color=color
            )
        plt.xlabel('LD1')
        plt.ylabel('Class')
        plt.title('LDA (k=1): LD1 vs class')
        plt.legend()
        plt.tight_layout()
        scatter_path = os.path.join(plots_dir, f'lda_k_{k}_scatter.png')
        plt.savefig(scatter_path)
     

    elif k == 2:
        for cls, color in zip(classes, colors):
            idx = (y == cls)
            plt.scatter(
                X_lda_k[idx, 0],
                X_lda_k[idx, 1],
                label=f'class_{cls}',
                edgecolors='k',
                color=color
            )
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.title(f'LDA (k={k}): LD1 vs LD2')
        plt.legend()
        plt.tight_layout()
        scatter_path = os.path.join(plots_dir, f'lda_k_{k}_scatter.png')
        plt.savefig(scatter_path)
       
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        for cls, color in zip(classes, colors):
            idx = (y == cls)
            ax.scatter(
                X_lda_k[idx, 0],
                X_lda_k[idx, 1],
                X_lda_k[idx, 2],
                label=f'class_{cls}',
                edgecolors='k',
                color=color
            )
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
        ax.set_title('LDA (k=3): LD1 vs LD2 vs LD3')
        ax.legend()
        plt.tight_layout()
        scatter_path = os.path.join(plots_dir, f'lda_k_{k}_scatter_3d.png')
        plt.savefig(scatter_path)
        

    plt.figure(figsize=(5, 4))
    components = np.arange(1, len(var_ratio) + 1)
    plt.bar(components, var_ratio)
    plt.xticks(components)
    plt.xlabel('Linear Discriminant')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'LDA Explained Variance Ratio (k={k})')
    plt.tight_layout()
  
    var_path = os.path.join(plots_dir, f'lda_k_{k}_var_ratio.png')
    plt.savefig(var_path)
  
    log_insight(f'LDA (k={k}) explained variance ratio: {var_ratio}')

    return X_lda_k

def main():
    data=data_import.load_data()
    data=data_preprocessing.data_preprocess(data)
    X=data.drop(columns='Air Quality')
    y=data['Air Quality']
    # feature_names= np.array(X.columns.tolist())
    # feature_names
    k=[1,2,3]
    for i in k:
        df=univeriate(i,X,y)
        df=feature_imp_score(i,X,y)
        df=pca(i,X,y)
        df=lda(i,X,y)

if __name__ == '__main__':
    main()