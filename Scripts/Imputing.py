'''
TODO: Description
'''

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class Imputing():
    '''
    This class performs data imputation for train-test pair 
    with separate imputers for categorical and numeric features.
    '''

    def __init__(self,):
        self.random_state = 20

    def impute(self, cat_imputer, noncat_imputer, train, test, cat_features_id):
        '''
        This function imputes missing values with Iterative Imputer.
        Separately for continuous and categorical features.
        '''

        # save a copy of categorical features with missing values
        train_mask = train.iloc[:, cat_features_id].copy()
        test_mask = test.iloc[:, cat_features_id].copy()

        # Fill missing values in categorical features with
        imp_most_frequent = SimpleImputer(
            missing_values=np.nan, strategy='most_frequent')
        train.iloc[:, cat_features_id] = imp_most_frequent.fit_transform(
            train.iloc[:, cat_features_id])
        test.iloc[:, cat_features_id] = imp_most_frequent.transform(
            test.iloc[:, cat_features_id])

        # Impute non-categorical missing values
        train = noncat_imputer.fit_transform(train)
        test = noncat_imputer.transform(test)

        train[:, cat_features_id] = train_mask
        test[:, cat_features_id] = test_mask

        # Impute categorical missing values
        train = cat_imputer.fit_transform(train)
        test = cat_imputer.transform(test)

        return train, test

    def process(self, cat_imputer, noncat_imputer, data, target, path, dataset_features, name,
                categorical_cols, download=False):
        '''This function is pipeline for imputation.'''

        # get train and test data
        X_train, X_test = data['train'].iloc[:, :-1], data['test'].iloc[:, :-1]
        # calculate index of categorical features
        categorical_features_index = [
            X_train.columns.get_loc(col) for col in categorical_cols]
        # Impute NAs
        X_train, X_test = self.impute(cat_imputer=cat_imputer, noncat_imputer=noncat_imputer, train=X_train,
                                      test=X_test, cat_features_id=categorical_features_index)

        # make train and test pandas datasets that include target column
        train_imputed = pd.DataFrame(data=X_train, columns=dataset_features)
        train_imputed[target] = data['train'].iloc[:, -1].values
        test_imputed = pd.DataFrame(data=X_test, columns=dataset_features)
        test_imputed[target] = data['test'].iloc[:, -1].values

        # report of resulting datasets
        print('Train shape:\t', train_imputed.shape)
        print('Train target:\n', train_imputed[target].value_counts(), end='')
        print('\n\n')
        print('Test shape:\t', test_imputed.shape)
        print('Test target:\n', test_imputed[target].value_counts(), end='')

        # download dataset
        if download:
            train_imputed.to_excel(f'{path}train_{name}.xlsx')
            test_imputed.to_excel(f'{path}test_{name}.xlsx')

        return train_imputed, test_imputed
