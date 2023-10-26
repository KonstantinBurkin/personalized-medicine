'''
TODO: Description
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class Preprocessing():
    '''
    Class for preprocessing:
        scaling
        calculating target
        spliting
    '''

    def __init__(self,):
        self.random_state = 20

    def get_combined_target_column(self, dataset):
        '''This function gets values of combined target.'''

        target_combined = (
            ((dataset['КОНЕЧНЫЕ ИСХОДЫ НАБЛЮДЕНИЯ']['Сердечно-сосудистая смерть'].astype(int) == 1) |
             (dataset['КОНЕЧНЫЕ ИСХОДЫ НАБЛЮДЕНИЯ']['Реинфаркт'].astype(int) == 1) |
             (dataset['КОНЕЧНЫЕ ИСХОДЫ НАБЛЮДЕНИЯ']['ОНМК'].astype(int) == 1) |
             (dataset['КОНТРОЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['Сердечно-сосудистая смерть'].astype(int) == 1) |
             (dataset['КОНТРОЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['Реинфаркт'].astype(int) == 1) |
             (dataset['КОНТРОЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['ОНМК'].astype(int) == 1)) &
            (dataset['ГОСПИТАЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['Смерть'] != 1)
        )*1

        mask_combined = (
            ((dataset['КОНЕЧНЫЕ ИСХОДЫ НАБЛЮДЕНИЯ']['Сердечно-сосудистая смерть'].astype(int) == -1) &
             (dataset['КОНЕЧНЫЕ ИСХОДЫ НАБЛЮДЕНИЯ']['Реинфаркт'].astype(int) == -1) &
             (dataset['КОНЕЧНЫЕ ИСХОДЫ НАБЛЮДЕНИЯ']['ОНМК'].astype(int) == -1) &
             (dataset['КОНТРОЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['Сердечно-сосудистая смерть'].astype(int) != 1) &
             (dataset['КОНТРОЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['Реинфаркт'].astype(int) != 1) &
             (dataset['КОНТРОЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['ОНМК'].astype(int) != 1)) |
            (dataset['ГОСПИТАЛЬНЫЕ КЛИНИЧЕСКИЕ ИСХОДЫ']['Смерть'] == 1)
        )
        target_combined[mask_combined] = np.nan

        return target_combined

    def scaler(self, train, test, continuous_features):
        '''This function returns train and test subsets with 
        scaled continuous features with Standard Scaler.'''

        scaler = StandardScaler()
        train[continuous_features] = scaler.fit_transform(
            train[continuous_features])
        test[continuous_features] = scaler.transform(test[continuous_features])

        return train, test

    def process(self, data, target, path, dataset_features, test_size, name,
                continuous_cols, save_before_split=False, download=False):
        '''This function is pipeline for dataset preprocessing.'''

        # make a copy of provided data for preprocessing
        dataset = data.copy()

        # in case we use combined target and NOT simply one of the features in dataset
        if target == ('target', 'combined'):
            # add new target column
            dataset[target] = self.get_combined_target_column(dataset)

        # exclude uncesessary features
        dataset = dataset[dataset_features + [target]]

        # drop patients with -1 in the target column
        dataset.dropna(axis=0, how='any', inplace=True)

        # replace missing values with NaN for the following imputation
        dataset.replace(-1, np.nan, inplace=True)

        if save_before_split:
            dataset.to_excel(f'{path}dataset_{name}.xlsx')

        # divide dataset into train and test
        X_train, X_test, y_train, y_test = \
            train_test_split(dataset[dataset_features],
                             dataset[target],
                             test_size=test_size,
                             random_state=self.random_state,
                             stratify=dataset[target],
                             shuffle=True)

        # scaling of dataset
        X_train, X_test = self.scaler(
            train=X_train, test=X_test, continuous_features=continuous_cols)

        # make train and test pandas datasets that include target column
        train_processed = pd.DataFrame(data=X_train, columns=dataset_features)
        train_processed[target] = y_train.values
        test_processed = pd.DataFrame(data=X_test, columns=dataset_features)
        test_processed[target] = y_test.values

        # report of resulting datasets
        print('Train shape:\t', train_processed.shape)
        print('Train target:\n',
              train_processed[target].value_counts(), end='')
        print('\n\n')
        print('Test shape:\t', test_processed.shape)
        print('Test target:\n', test_processed[target].value_counts(), end='')

        # download dataset
        if download:
            train_processed.to_excel(f'{path}train_{name}.xlsx')
            test_processed.to_excel(f'{path}test_{name}.xlsx')

        return train_processed, test_processed
