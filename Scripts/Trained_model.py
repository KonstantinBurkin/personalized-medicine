# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
# 
import numpy as np
import pandas as pd
import ast
import pickle

with open('../Optimisation files/Combined target/grid.pickle', 'rb') as f:
    grid = pickle.load(f)


class Trained_model():
    '''
    Class for getting train and test data and model instance. 
    '''

    def __init__(self,
                 targets: list, #  = ['combined'] cardiovascular death, revascularization, combined
                 datasets: list, #  = ['a'] a, b, c, abc
                 subsets: list, #  = ['Biomarkers + Clinical'] Biomarkers, Clinical, Biomarkers + Clinical
                 models: list, #  = ['randomforest'] svm, knn, LogisticRegression, catboost, randomforest
                 selectors: list, #  = ['SHAP'] RF_feature_importance, SFS/RFE, SHAP
                 N: int = 10,
                 random_state: int = 20):

        self.targets = targets
        self.datasets = datasets
        self.subsets = subsets
        self.models = models
        self.random_state = random_state
        self.selectors = selectors
        self.N = N
        self.save=False

        clinical_features = list(map(tuple, pd.read_excel('../Raw data/Clinical features.xlsx', index_col=0, header=0).values))
        biomarkers_a = list(map(tuple, pd.read_excel('../Raw data/biomarkers_a.xlsx', index_col=0, header=0).values.tolist()))
        biomarkers_b = list(map(tuple, pd.read_excel('../Raw data/biomarkers_b.xlsx', index_col=0, header=0).values.tolist()))
        biomarkers_c = list(map(tuple, pd.read_excel('../Raw data/biomarkers_c.xlsx', index_col=0, header=0).values.tolist()))
        clinical_features = list(map(str, clinical_features))
        biomarkers_a = list(map(str, biomarkers_a))
        biomarkers_b = list(map(str, biomarkers_b))
        biomarkers_c = list(map(str, biomarkers_c))


        self.columns = {'a'    : np.array([clinical_features, biomarkers_a]),
                        'b'    : np.array([clinical_features, biomarkers_b]),
                        'c'    : np.array([clinical_features, biomarkers_c]),
                        'd'    : np.array([clinical_features]),
                        'abcd' : np.array([clinical_features]) }
        
        self.clinical = clinical_features

        

    def get_train_test(self, link_train, link_test, dataset, subset, drop_col,):
        '''
        Function for getting train and test for specified dataset and subset
        '''
        

        if subset == 'Biomarkers + Clinical':
            features = self.columns[dataset][0] + self.columns[dataset][1]
        elif subset == 'Biomarkers':
            features = self.columns[dataset][1]
        else:
            features = self.columns[dataset][0]

        target = "('target', 'combined')"

        # использую названия признаков как строки мб стоит исправить
        X_train = pd.read_excel(link_train, header=[0], usecols=features)
        y_train = pd.read_excel(link_train, header=[0], usecols=[target])
        X_test  = pd.read_excel(link_test, header=[0], usecols=features)
        y_test  = pd.read_excel(link_test, header=[0], usecols=[target])

        # if drop_col == 'drop':
        #     # drop GRACE feature
        #     if subset != 'Biomarkers':
        #         X_train.drop("('ХАРАКТЕРИСТИКА ОИМ', 'Риск GRACE, баллы')", axis=1, inplace=True)
        #         X_test.drop("('ХАРАКТЕРИСТИКА ОИМ', 'Риск GRACE, баллы')", axis=1, inplace=True)
        # elif drop_col == 'only':
        #     # leave only GRACE feature from all clinical features
        #     if subset != 'Biomarkers':
        #         if "('ХАРАКТЕРИСТИКА ОИМ', 'Риск GRACE, баллы')" in self.clinical:
        #             self.clinical.remove("('ХАРАКТЕРИСТИКА ОИМ', 'Риск GRACE, баллы')")
        #         X_train.drop(self.clinical, axis=1, inplace=True)
        #         X_test.drop(self.clinical, axis=1, inplace=True)
        if drop_col == None:
            # leave all clinical features
            pass
        else:
            if subset != 'Biomarkers':
                # if "('ХАРАКТЕРИСТИКА ОИМ', 'Риск GRACE, баллы')" in self.clinical:
                #     self.clinical.remove("('ХАРАКТЕРИСТИКА ОИМ', 'Риск GRACE, баллы')")
                X_train.drop(drop_col, axis=1, inplace=True)
                X_test.drop(drop_col, axis=1, inplace=True)

        

        return X_train, X_test, y_train, y_test


    def get_model_instance(self, model, target, subset, dataset, X_train, y_train):
        '''
        Get trained model instance
        '''

        path = f"../Optimisation files/{target}/{subset} {dataset.upper()}/all/"
        # read file with tuned parameters
        try:
            optimisation_file = pd.read_excel(f'{path}{model}_optimisation.xlsx', header=[0])
        except:
            print(f"There is no file in {path} \n")

        try:
            params = optimisation_file[optimisation_file['rank_test_roc_auc']==1][["params"]].iloc[0]
            params = ast.literal_eval(params[0])
            if model == 'randomforest':
                model_instance = RandomForestClassifier(**params)
            elif model == 'knn':
                model_instance = KNeighborsClassifier(**params)
            elif model == 'svm':
                model_instance = SVC(**params)
            elif model == 'LogisticRegression':
                model_instance = LogisticRegression(**params)
            elif model == 'adaboost':
                model_instance = AdaBoostClassifier(**params)
        except:
            # catboost has additional parameters
            params = optimisation_file['params'][0]
            params = ast.literal_eval(params)
            model_instance = CatBoostClassifier(**grid['CB'][1], **params)

        model_instance.fit(X_train, y_train)

        return model_instance