import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt


class Visualisation:
    def __init__(self, data):
        sns.set(rc={'figure.figsize': (11.7, 6.27)})
        self.palette = "cool"
        self.color = "darkblue"
        self.data = data
        self.target = data.target

    def plot_relationships(self, df, col, target, hue=None):
        '''creates a plot to see relationship between target and variables'''
        if df[col].dtype == 'int64':
            sns.lineplot(x=col, y=target, hue=hue, data=df, palette=self.palette)
        else:
            sorted_values = df.groupby(col)[target].median().sort_values()
            sns.boxplot(x=col, y=target, data=df, order=list(sorted_values.index),
                        palette=self.palette)

    def two_way_table(self, df, col, target, margins=True):
        '''creates a table to see relationship between two numerical variables'''
        table = pd.crosstab(df[col], df[target], margins=margins, margins_name="Total")
        return table

    def plot_distribution(self, df, col):
        '''distribution plot for target'''
        plt.subplot(1, 2, 1)
        sns.distplot(df[col], color=self.color, kde=True, bins=20, label='train')
        plt.subplot(1, 2, 2)
        plt.title('Normal')
        sns.distplot(df[col], color=self.color, kde=True, fit=st.norm)
