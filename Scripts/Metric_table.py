# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# 
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as TP_rate                          
from sklearn.metrics import roc_auc_score as roc_auc
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score as recall
from sklearn.metrics import average_precision_score
import pickle

with open('../Optimisation files/Combined target/grid.pickle', 'rb') as f:
    grid = pickle.load(f)



class Metric_table():
    '''
    This class returns matric table of 5 models (LR, kNN, RF, SVM, CatBoost).
    Optimisation folder with 5 optimisation files must be specified.
    Train and test datasets should be provided. 
    '''

    def metric_table(self, path, X_train, y_train, X_test, y_test):  #, X_train=X_train, y_train=y_train
        '''
        This function provides scores for gridsearch F1-score and metrics for test dataset
        '''
        # read gridsearch tables
        randomforest_optimisation = pd.read_excel(f'{path}randomforest_optimisation.xlsx', header=[0]) #/content/  ./imp_feat
        svm_optimisation = pd.read_excel(f'{path}svm_optimisation.xlsx', header=[0])
        knn_optimisation = pd.read_excel(f'{path}knn_optimisation.xlsx', header=[0])
        LogisticRegression_optimisation = pd.read_excel(f'{path}LogisticRegression_optimisation.xlsx', header=[0])
        adaboost_optimisation = pd.read_excel(f'{path}adaboost_optimisation.xlsx', header=[0])
        catboost_optimisation = pd.read_excel(f'{path}catboost_optimisation.xlsx', header=[0])

        params = randomforest_optimisation[randomforest_optimisation['rank_test_mcc']==1][["params"]].iloc[0]
        params = ast.literal_eval(params[0])
        random_forest_model = RandomForestClassifier(**params)
        # 
        params = svm_optimisation[svm_optimisation['rank_test_mcc']==1][["params"]].iloc[0]
        params = ast.literal_eval(params[0])
        SVM_model = SVC(**params)
        # 
        # params = nn_optimisation[nn_optimisation['rank_test_mcc']==1][["params"]].iloc[0]
        # params = ast.literal_eval(params[0])
        # newral_network_model = MLPClassifier(**params)
        # 
        params = knn_optimisation[knn_optimisation['rank_test_mcc']==1][["params"]].iloc[0]
        params = ast.literal_eval(params[0])
        knn_model = KNeighborsClassifier(**params)
        # 
        params = LogisticRegression_optimisation[LogisticRegression_optimisation['rank_test_mcc']==1][["params"]].iloc[0]
        params = ast.literal_eval(params[0])
        LR_model = LogisticRegression(**params)
        # 
        params = adaboost_optimisation[adaboost_optimisation['rank_test_mcc']==1][["params"]].iloc[0]
        params = eval(params[0])
        AB_model = AdaBoostClassifier(**params)
        # 
        params = catboost_optimisation['params'][0]
        params = ast.literal_eval(params)
        catboost_model = CatBoostClassifier(**grid['CB'][1], **params)

        models = [
        random_forest_model,
        SVM_model,
        LR_model,
        knn_model,
        AB_model,
        catboost_model
        ]

        mcc_score, f1_score,f2_score, accuracy_score, TP_rate_score, recall_score, auc_precision_recall, roc_auc_score= [], [], [], [], [], [], [], []
        tn, fp, fn, tp = [], [], [], []

        for model in models:
            model.fit(X_train, y_train)
            forecast = model.predict(X_test)
            forecast_proba = model.predict_proba(X_test)

            mcc_score.append(mcc(y_test, forecast))                                   # MCC
            f1_score.append(f1(y_test, forecast))                                       # F1
            # f2_score.append(f2_func(y_test, forecast))                                  # F1
            accuracy_score.append(accuracy(y_test, forecast))                           # Accuracy  
            TP_rate_score.append(TP_rate(y_test, forecast))                             # TP rate   tp / (tp + fp)
            recall_score.append(recall(y_test, forecast))                               # TN rate
            auc_precision_recall.append(average_precision_score(y_test, forecast_proba[:,1]))      # PR AUC
            roc_auc_score.append(roc_auc(y_test, forecast_proba[:,1]))                       # ROC AUC
            tn.append(confusion_matrix(y_test, forecast).ravel()[0])                  # number of true negative
            fp.append(confusion_matrix(y_test, forecast).ravel()[1])                  # number of false positive
            fn.append(confusion_matrix(y_test, forecast).ravel()[2])                  # number of false negative
            tp.append(confusion_matrix(y_test, forecast).ravel()[3])                  # number of true positive

        # create matrix table 
        metrics_table = pd.DataFrame(columns=pd.MultiIndex.from_product([["MCC, train set, cv=5", "ROC_AUC, train set, cv=5"],["mean", 'std']]))
        metrics_table[("Scores on the test set","MCC")] = mcc_score
        metrics_table[("Scores on the test set","F1")] = f1_score
        # metrics_table[("Scores on the test set","F2")] = f2_score
        metrics_table[("Scores on the test set","Accuracy")] = accuracy_score
        metrics_table[("Scores on the test set","Precision")] = TP_rate_score
        metrics_table[("Scores on the test set","Recall")] = recall_score
        metrics_table[("Scores on the test set","PR_AUC")] = auc_precision_recall
        metrics_table[("Scores on the test set","ROC_AUC")] = roc_auc_score
        metrics_table[("Confusion matrix","TN")] = tn
        metrics_table[("Confusion matrix","FP")] = fp
        metrics_table[("Confusion matrix","FN")] = fn
        metrics_table[("Confusion matrix","TP")] = tp

        # modify the rows names
        metrics_table.index = [
                    "Random Forest",
                    "SVM",
                    "Logistic Regression",
                    "KNN",
                    "adaBoost",
                    "CatBoost"
                    ]



        # add cross validated F2 scores on the train set
        mean = []
        std = []
        mean_test,std_test = randomforest_optimisation[randomforest_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        randomforest_optimisation[randomforest_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        mean.append(mean_test); std.append(std_test)
        mean_test,std_test = svm_optimisation[svm_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        mean.append(mean_test); std.append(std_test)
        # mean_test,std_test = nn_optimisation[nn_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        # mean.append(mean_test); std.append(std_test)
        mean_test,std_test = LogisticRegression_optimisation[LogisticRegression_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        mean.append(mean_test); std.append(std_test)
        mean_test,std_test = knn_optimisation[knn_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        mean.append(mean_test); std.append(std_test)
        mean_test,std_test = adaboost_optimisation[adaboost_optimisation['rank_test_mcc']==1][["mean_test_mcc","std_test_mcc"]].iloc[0]
        mean.append(mean_test); std.append(std_test)
        mean_test,std_test = catboost_optimisation[['mean_test_mcc', 'std_test_mcc']].iloc[catboost_optimisation.shape[0]-1]
        mean.append(mean_test); std.append(std_test)
        

        metrics_table[("MCC, train set, cv=5","mean")] = mean
        metrics_table[("MCC, train set, cv=5","std")] = std

        # add cross validated F2 scores on the train set
        mean_roc_auc = []
        std_roc_auc = []
        mean_test_roc_auc,std_test_roc_auc = randomforest_optimisation[randomforest_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        randomforest_optimisation[randomforest_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)
        mean_test_roc_auc,std_test_roc_auc = svm_optimisation[svm_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)
        # mean_test_roc_auc,std_test_roc_auc = nn_optimisation[nn_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        # mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)
        mean_test_roc_auc,std_test_roc_auc = LogisticRegression_optimisation[LogisticRegression_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)
        mean_test_roc_auc,std_test_roc_auc = knn_optimisation[knn_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)
        mean_test_roc_auc,std_test_roc_auc = adaboost_optimisation[adaboost_optimisation['rank_test_mcc']==1][["mean_test_roc_auc","std_test_roc_auc"]].iloc[0]
        mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)
        mean_test_roc_auc,std_test_roc_auc = catboost_optimisation[["mean_test_roc_auc","std_test_roc_auc"]].iloc[catboost_optimisation.shape[0]-1]
        mean_roc_auc.append(mean_test_roc_auc); std_roc_auc.append(std_test_roc_auc)

        metrics_table[("ROC_AUC, train set, cv=5","mean")] = mean_roc_auc
        metrics_table[("ROC_AUC, train set, cv=5","std")] = std_roc_auc

        return metrics_table