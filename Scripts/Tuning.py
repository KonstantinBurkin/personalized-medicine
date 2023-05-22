# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

with open('../Optimisation files/Combined target/grid.pickle', 'rb') as f:
    grid = pickle.load(f)


class Tuner():
    '''
    Class for general tuning. 
    It creates a pipeline for tunning 5 models (LR, kNN, RF, SVM, CatBoost).
    It saves tuned parameters in excel-files.
    '''

    def __init__(self, catboost_score: str, score):
        self.score = score
        self.catboost_score = catboost_score
        self.random_state = 20
        self.cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
        
        self.logistic_regression = True
        self.knn = False
        self.bayes = False
        self.random_forest = True
        self.svm = True
        self.catboost = True
        self.adaboost = True


    def tuning(self, path, X_train, X_test, y_train, y_test):
        if self.logistic_regression:
            self.logistic_regression_tuning(self.score, self.cross_validation, path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,)
        if self.knn:
            self.knn_tuning(self.score, self.cross_validation, path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,)     
        if self.random_forest:
            self.random_forest_tuning(self.score, self.cross_validation, path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,)     
        if self.svm:
            self.svm_tuning(self.score, self.cross_validation, path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,)     
        if self.catboost:
            self.catboost_tuning(self.catboost_score, self.score, self.cross_validation,  path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,)  
        if self.adaboost:
            self.adaboost_tuning(self.score, self.cross_validation, path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,)  


    def adaboost_tuning(self, score, cross_validation, path, X_train, X_test, y_train, y_test):
      
        model = AdaBoostClassifier(base_estimator=None, random_state=self.random_state)

        # calibrate hyper-parameters: perform gridsearch with cross-validation
        clf = GridSearchCV(
                          estimator = model, 
                          param_grid = grid['AB'],
                          scoring = score,    
                          refit = list(score.keys())[0],    
                          cv = cross_validation,
                          n_jobs = -1
                          )              
        clf.fit(X_train, y_train)
        
        # Save the results
        model = clf.best_estimator_
        # write down optimisation parameters
        optimisation_table = pd.DataFrame(clf.cv_results_)
        # save optimisation table
        optimisation_table.to_excel(f'{path}adaboost_optimisation.xlsx') 
        # # clear_output()


    def logistic_regression_tuning(self, score, cross_validation, path, X_train, X_test, y_train, y_test):
      
        model = LogisticRegression(random_state=self.random_state)

        # calibrate hyper-parameters: perform gridsearch with cross-validation
        clf = GridSearchCV(
                          estimator = model, 
                          param_grid = grid['LR'],
                          scoring = score,    
                          refit = list(score.keys())[0],    
                          cv = cross_validation,
                          n_jobs = -1
                          )              
        clf.fit(X_train, y_train)
        
        # Save the results
        model = clf.best_estimator_
        # write down optimisation parameters
        optimisation_table = pd.DataFrame(clf.cv_results_)
        # add roc_auc fCV values
        # optimisation_table['roc_auc'] = str(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['mean_test_roc_auc'] = np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['std_test_roc_auc'] = np.std(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # save optimisation parameters and CV values of F-metric and ROC_AUC
        optimisation_table.to_excel(f'{path}LogisticRegression_optimisation.xlsx') 
        # # clear_output()


    def knn_tuning(self, score, cross_validation, path, X_train, X_test, y_train, y_test):
        
        model = KNeighborsClassifier()
        # calibrate hyper-parameters: perform gridsearch with cross-validation = 5 
        clf = GridSearchCV(
                          estimator=model, 
                          param_grid=grid['KNN'],
                          scoring=score,
                          refit = list(score.keys())[0], 
                          cv=cross_validation,
                          n_jobs=-1
                          )              
        clf.fit(X_train, y_train)
        model = clf.best_estimator_

        # write down optimisation parameters
        optimisation_table = pd.DataFrame(clf.cv_results_)
        # add roc_auc fCV values
        # optimisation_table['roc_auc'] = str(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['mean_test_roc_auc'] = np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['std_test_roc_auc'] = np.std(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # save optimisation parameters and CV values of F-metric and ROC_AUC 
        optimisation_table.to_excel(f'{path}knn_optimisation.xlsx')
        # clear_output()


    def random_forest_tuning(self, score, cross_validation, path, X_train, X_test, y_train, y_test):
      
        model = RandomForestClassifier(random_state=self.random_state)

        # calibrate hyper-parameters: perform gridsearch with cross-validation = 5 
        clf = GridSearchCV(
                          estimator=model, 
                          param_grid=grid['RF'],
                          scoring=score,  
                          refit = list(score.keys())[0], 
                          cv=cross_validation,
                          n_jobs=-1
                          )              
        clf.fit(X_train, y_train)
        model = clf.best_estimator_

        # write down optimisation parameters
        optimisation_table = pd.DataFrame(clf.cv_results_)
        # add roc_auc fCV values
        # optimisation_table['roc_auc'] = str(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['mean_test_roc_auc'] = np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['std_test_roc_auc'] = np.std(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # save optimisation parameters and CV values of F-metric and ROC_AUC
        optimisation_table.to_excel(f'{path}randomforest_optimisation.xlsx')
        # clear_output()


    def svm_tuning(self, score, cross_validation, path, X_train, X_test, y_train, y_test):
  
        model = SVC(random_state=self.random_state)

        # calibrate hyper-parameters: perform gridsearch with cross-validation = 5 
        clf = GridSearchCV(
                          estimator=model, 
                          param_grid=grid['SVM'],
                          scoring=score,  
                          refit = list(score.keys())[0], 
                          cv=cross_validation,
                          n_jobs=-1
                          )              
        clf.fit(X_train, y_train)
        model = clf.best_estimator_

        # write down optimisation parameters
        optimisation_table = pd.DataFrame(clf.cv_results_)

        # add roc_auc fCV values
        # optimisation_table['roc_auc'] = str(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['mean_test_roc_auc'] = np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # optimisation_table['std_test_roc_auc'] = np.std(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc'))
        # save optimisation parameters and CV values of F-metric and ROC_AUC
        optimisation_table.to_excel(f'{path}svm_optimisation.xlsx')
        # clear_output()
        

    def catboost_tuning(self, catboost_score, score, cross_validation, path, X_train, X_test, y_train, y_test):

        np.random.seed(10)
        catboost = CatBoostClassifier(
                                    eval_metric=catboost_score,
                                    # verbose=False,
                                    early_stopping_rounds=100,
                                    task_type="CPU",
                                    loss_function='Logloss',
                                    logging_level='Silent',
                                    iterations = 500,
                                    random_seed=10)


        grid_res = catboost.grid_search(grid['CB'][0],
                                        X_train,
                                        y_train,
                                        cv=cross_validation,
                                        search_by_train_test_split=False,
                                        calc_cv_statistics=True,
                                        refit=True,
                                        shuffle=True,
                                        partition_random_seed=10,
                                        verbose=False,
                                        stratified=True)

        # write down optimisation parameters
        cv_results = pd.DataFrame(grid_res['cv_results'])
        cv_results['params'] = 0
        cv_results['params'] = str(grid_res['params'])

        # add roc_auc fCV values
        cvs = cross_val_score(catboost, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=20), scoring='roc_auc')
        cv_results['roc_auc'] = str(cvs)
        cv_results['mean_test_roc_auc'] = np.mean(cvs)
        cv_results['std_test_roc_auc'] = np.std(cvs)

        # add MCC fCV values
        cvs = cross_val_score(catboost, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=20), scoring=score['mcc'])
        cv_results['mcc'] = str(cvs)
        cv_results['mean_test_mcc'] = np.mean(cvs)
        cv_results['std_test_mcc'] = np.std(cvs)
        # save optimisation parameters and CV values of F-metric and ROC_AUC
        cv_results.to_excel(f'{path}catboost_optimisation.xlsx')
        # clear_output()



from Metric_table import Metric_table
from Trained_model import Trained_model

class Tunning(Trained_model, Tuner, Metric_table):
    '''
    Class for tuning.
    # Traget and dataset should be specified.
    # Tuned parameters of model are read, and model is retrained.
    # Chosen selectors are used to determine top N features. 
    '''

    def __init__(self, targets, datasets, subsets, models, selectors, catboost_score, score, drop_col, folder):
        Trained_model.__init__(self, targets, datasets, subsets, models, selectors,)
        Tuner.__init__(self, catboost_score, score,)
        self.drop_col = drop_col
        self.folder = folder

    def tune(self):
        '''
        Function for feature selection.
        Traget and dataset should be specified.
        Tuned parameters of model are read, and model is retrained.
        Chosen selectors are used to determine top N features. 
        '''

        for target in tqdm(self.targets, desc='Targets loop', leave=True):

            for dataset in tqdm(self.datasets, leave=False, desc='datasets'):
                # links to preprocessed 'train' and 'test' data
                link_train = f'../Preprocessed Data/{target}/Imputed data/train_{dataset}.xlsx'
                link_test = f'../Preprocessed Data/{target}/Imputed data/test_{dataset}.xlsx'

                for subset in tqdm(self.subsets, leave=False, desc='subsets'):

                    # tuning cycle with preselected top features
                    if self.selectors[0] != '':
                        for selector in self.selectors:
                            # SHAP
                            if selector == 'preselected':
                                pass
                            else:
                                try:
                                    top = list(pd.read_excel(f'./HSE project/Feature selection/{target}/{subset} {dataset.upper()}/{selector}/randomforest_values.xlsx', header=[0], index_col=[0]).sort_values(by='values', ascending=False)['features'][:self.N].values)
                                    top = list(pd.read_excel(f'./HSE project/Feature selection/{target}/{subset} {dataset.upper()}/{selector}/randomforest_values.xlsx', header=[0], index_col=[0]).sort_values(by='values', ascending=False)['features'][:self.N].values)
                                except:
                                    top = list(pd.read_excel(f'./HSE project/Feature selection/{target}/{subset} {dataset.upper()}/{selector}/values.xlsx', header=[0], index_col=[0]).sort_values(by='values', ascending=False)['features'][:self.N].values)
                            
                            # upload 'train' and 'test' data
                            X_train, X_test, y_train, y_test = self.get_train_test(link_train, link_test, dataset, subset, self.drop_col)
                            
                            if selector != 'preselected':
                                X_train = X_train[top]
                                X_test = X_test[top]
                                optimisation_path = f"../Optimisation files/{target}/{subset} {dataset.upper()}/top/{selector}/"
                                
                            optimisation_path = f"../Optimisation files/{target}/{subset} {dataset.upper()}/top//"

                            self.tuning(path = optimisation_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                            metrics_table = self.metric_table(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, path=optimisation_path, )
                            if self.save:
                                metrics_table.to_excel(f'{optimisation_path}metrics_table.xlsx')
                    
                    # general tuning with all features
                    else:
                        X_train, X_test, y_train, y_test = self.get_train_test(link_train, link_test, dataset, subset, self.drop_col) 

                        # if self.drop_col != None:
                        #     optimisation_path = f"../Optimisation files/{target}/{subset} {dataset.upper()}/{'all' if 'GRACE' in self.drop_col else 'grace'}/"
                        # else:
                        optimisation_path = f"../Optimisation files/{target}/{subset} {dataset.upper()}/{self.folder}/"

                        self.tuning(path = optimisation_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                        metrics_table = self.metric_table(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, path=optimisation_path, )
                        if self.save:
                            metrics_table.to_excel(f'{optimisation_path}metrics_table.xlsx')
