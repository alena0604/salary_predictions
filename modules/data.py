import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, train_feature_file, train_target_file, test_feature_file, target, id_cols):
        '''creates train and test dataframe'''
        self.id_cols = list(id_cols)
        self.target = target
        self.train_df = self._get_train_df(train_feature_file, train_target_file)
        self.test_df = self._get_test_df(test_feature_file)
        self.test_train_df = self._get_concat_dfs(train_feature_file, test_feature_file)

    def _get_train_df(self, train_feature_file, train_target_file):
        '''loads and merges training data'''
        train_feature_df = self._load_data(train_feature_file)
        train_target_df = self._load_data(train_target_file)
        train_df = self._merge_dfs(train_feature_df, train_target_df)
        return train_df

    def _get_test_df(self, test_feature_file):
        '''loads test data'''
        test_df = self._load_data(test_feature_file)
        return test_df

    def _load_data(self, file):
        return pd.read_csv(file)

    def _merge_dfs(self, df1, df2, key=None, left_index=False, right_index=False):
        return pd.merge(left=df1, right=df2, how='inner', on=key,
                        left_index=left_index, right_index=right_index)

    def _concatenate_dfs(self, df1, df2):
        return pd.concat([df1, df2])

    def _get_concat_dfs(self, train_feature_file, test_feature_file):
        train_feature_df = self._load_data(train_feature_file)
        test_feature_df = self._load_data(test_feature_file)
        concat_df = self._concatenate_dfs(train_feature_df, test_feature_df)
        return concat_df

    def dataset_info(self, file):
        '''prints main information'''
        df = self._load_data(file)
        print('\n{0:*^80}'.format(' Reading from the file {0} '.format(file)))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.columns)
        print('\n{0:*^80}\n'.format(' Description of quantitative columns'))
        print(df.describe(include=[np.number]))
        print('\n{0:*^80}\n'.format(' Description of categorical columns'))
        print(df.describe(include=['O']))
