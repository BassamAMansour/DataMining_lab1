import pandas as pandas
import matplotlib.pyplot as plot


def plot_data():
    data = pandas.read_csv('lab2.csv', index_col='ID').groupby('Diagnosis')
    show_histogram(data)
    show_boxplot(data)
    show_scatter(data)
    show_line(data)
    show_pearson_corr(data)


def show_histogram(data):
    axes = data.plot.hist(bins=5, alpha=0.4)
    rename_titles(axes, data)
    plot.show()


def show_boxplot(data):
    axes = data.plot.box()
    rename_titles(axes, data)
    plot.show()


def show_scatter(data):
    axes = data.plot.scatter(x='Area', y='Smoothness')
    rename_titles(axes, data)
    plot.show()


def show_line(data):
    axes = data.plot.line()
    rename_titles(axes, data)
    plot.show()


def show_pearson_corr(data):
    corr = data.corr(method='pearson')
    plot.imshow(data.corr(method='pearson'))
    print(corr)
    plot.show()


def rename_titles(axes, data):
    for i, (group_name, group) in enumerate(data):
        axes[i].set_title(group_name)


if __name__ == "__main__":
    plot_data()
