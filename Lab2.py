import pandas as pandas
import matplotlib.pyplot as plot
from sklearn import preprocessing


def plot_data():
    data = pandas.read_csv('lab2.csv', index_col='ID')
    data = normalize(data).groupby('Diagnosis')
    show_histogram(data)
    show_boxplot(data)
    show_scatter(data)
    show_line(data)
    show_pearson_corr(data)


def normalize(data):
    normalized = preprocessing.MinMaxScaler().fit_transform(data.iloc[:, 1:])
    norm_df = pandas.DataFrame(data=normalized,
                               index=data.index,
                               columns=data.columns[1:])
    return norm_df.join(data.iloc[:, 0:1])


def show_histogram(data):
    axes = data.plot.hist(bins=5, alpha=0.4)
    rename_titles(axes, data)
    plot.show()


def show_boxplot(data):
    axes = data.plot.box()
    rename_titles(axes, data)
    plot.show()


def show_scatter(data):
    axes = data.plot.scatter(x='Largest Area', y='Largest Smoothness')
    rename_titles(axes, data)
    plot.show()


def show_line(data):
    axes = data.plot.line()
    rename_titles(axes, data)
    plot.show()


def show_pearson_corr(data):
    plot.imshow(data.corr(method='pearson'))
    plot.show()


def rename_titles(axes, data):
    for i, (group_name, group) in enumerate(data):
        axes[i].set_title(group_name)


if __name__ == "__main__":
    plot_data()
