import pandas as pd


class FeatureEngineering:
    def __init__(self, data):
        '''initializes class and creates groupby object for data'''
        self.data = data
        self.target = data.target
        self.id_cols = data.id_cols
        self.column_bins = 'yearsExperience'
        self.train_df = self.apply_experience_bins(self.data.train_df, self.column_bins)
        self.test_df = self.apply_experience_bins(self.data.test_df, self.column_bins)

    def get_experience_bins(self, column_value):
        '''creates bins from experience column'''
        if column_value < 2:
            return '0-1'
        elif 1 < column_value < 3:
            return '2'
        elif 3 <= column_value < 5:
            return '3-5'
        elif 5 <= column_value < 10:
            return '5+'
        else:
            return '10+'

    def apply_experience_bins(self, df, col):
        '''applies experience bins method'''
        df[col + '_bins'] = df[col].apply(self.get_experience_bins)
        return df

    def get_agg_features(self, df, col, target):
        '''creates aggregation features'''
        target_aggregation = {target: ['min', 'max', 'std', 'mean', 'median', 'skew']}
        if isinstance(col, list):
            df_agg = df.groupby(col).agg(target_aggregation)
        else:
            df_agg = df.groupby([col]).agg(target_aggregation)
        return df_agg

    def merge_agg_cols(self, df, col):
        '''merges aggregation features'''
        train_df = self.clean_data(self.train_df)
        feature_df = self.get_agg_features(train_df, col, self.target)
        if isinstance(col, list):
            feature_df.reset_index(inplace=True)
            df = pd.merge(df, feature_df, on=col, how='left')
        else:
            feature_df.columns = pd.Index([col + '_' + e[1].upper() for e in feature_df.columns])
            feature_df.reset_index(inplace=True)
            df = pd.merge(df, feature_df, on=col, how='left')
        return df

    def update_dfs(self, df):
        '''creates aggregation features'''
        df_cat = self.get_categorical_col(df)
        for col in df_cat:
            df = self.merge_agg_cols(df, col)
        return df

    def clean_data(self, df):
        '''removes rows that contain salary <= 0 or duplicate job IDs'''
        df = df.drop_duplicates(subset='jobId')
        df = df[df.salary > 0]
        return df

    def get_dummies(self, df):
        '''converts categorical variable into dummy variables'''
        df_cat = self.get_categorical_col(df)
        return pd.get_dummies(df, columns=df_cat)

    def get_categorical_col(self, df):
        '''gets a list of categorical variables'''
        df_cat = [f for f in df.columns if df[f].dtype == 'object' and f not in self.id_cols]
        return df_cat
