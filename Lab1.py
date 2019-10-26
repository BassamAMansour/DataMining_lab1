import pandas as pandas
import matplotlib.pyplot as plot
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_iris


def plot_data():
    ###### q1
    df = combine_data()
    data = df.groupby('ClassName')

    ###### q2
    show_histogram(data)

    ###### q3
    # show_boxplot(data)

    ###### q4
    # print df.corr(method='pearson')
    # show_pearson_corr(data)

    # minmax_norm(df)
    # zscore_norm(df)
    # getpca(df)
    # getkbest(df)


def minmax_norm(df):
    minmax = preprocessing.MinMaxScaler().fit_transform(df)
    df_minmax = pandas.DataFrame(data=minmax[0:, 0:],
                                 index=df.index,
                                 columns=list(df))
    show_histogram(df_minmax.groupby('ClassName'))


def zscore_norm(df):
    zscore = preprocessing.StandardScaler().fit_transform(df)
    df_zscore = pandas.DataFrame(data=zscore[0:, 0:],
                                 index=df.index,
                                 columns=list(df))
    show_histogram(df_zscore.groupby('ClassName'))


def getpca(df):
    zscore = preprocessing.StandardScaler().fit_transform(df)
    df_zscore = pandas.DataFrame(data=zscore[0:, 0:],
                                 index=df.index,
                                 columns=list(df))
    pca = PCA(n_components=3)
    pca.fit(df_zscore)
    df_pca = pandas.DataFrame(pca.components_)
    print(df_pca.corr(method='pearson'))
    print(pca.explained_variance_ratio_)


def getkbest(df):
    minmax = preprocessing.MinMaxScaler().fit_transform(df)
    select_k_best_classifier = SelectKBest(chi2, k=5).fit_transform(minmax, df.index.values)
    df_kbest = pandas.DataFrame(data=select_k_best_classifier[0:, 0:],
                                index=df.index,
                                columns=[0, 1, 2, 3, 4])
    show_histogram(df_kbest.groupby('ClassName'))


def show_histogram(data):
    axes = data.plot.hist(bins=5, alpha=0.4)
    rename_titles(axes, data)
    plot.show()


def show_boxplot(data):
    axes = data.plot.box()
    rename_titles(axes, data)
    plot.show()


def show_pearson_corr(data):
    plot.imshow(data.corr(method='pearson'))
    plot.show()


def rename_titles(axes, data):
    for i, (group_name, group) in enumerate(data):
        axes[i].set_title(group_name)


def combine_data():
    df1 = pandas.read_csv('data.csv', index_col='ClassName')
    df2 = pandas.read_csv('test.csv', index_col='ClassName')
    cols = [19]
    df1.drop(df1.columns[cols], axis=1, inplace=True)
    df2.drop(df2.columns[cols], axis=1, inplace=True)
    return pandas.concat([df1, df2])


if __name__ == "__main__":
    plot_data()
