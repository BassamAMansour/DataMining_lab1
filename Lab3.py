import numpy as np
import pandas as pandas
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


def main():
    data_frame = pandas.read_csv('lab2.csv', index_col='ID')
    zscore_normalized_data = normalize(data_frame, 'zscore')

    KNN(zscore_normalized_data, tune_KNN(zscore_normalized_data))
    linear_SVM(zscore_normalized_data, tune_linear_svm(zscore_normalized_data))
    logistic_regression(zscore_normalized_data, tune_logistic_regression(zscore_normalized_data))

    c, gamma = tune_non_linear_SVM(zscore_normalized_data)
    non_linear_SVM(zscore_normalized_data, c, gamma)


def KNN(data, n):
    X_train, X_test, y_train, y_test = split_data(data)

    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train.values.ravel())
    y_predict = knn.predict(X_test)

    print_metrics(f'KNN. N = {n}.', y_test, y_predict)


def tune_KNN(data):
    X_train, X_test, y_train, y_test = split_data(data)

    param_grid = {'n_neighbors': np.arange(1, 100)}
    knn = KNeighborsClassifier()
    gs = GridSearchCV(knn, param_grid, cv=10)
    gs.fit(X_train, y_train.values.ravel())

    return gs.best_params_['n_neighbors']


def linear_SVM(data, c):
    X_train, X_test, y_train, y_test = split_data(data)

    lsvm = LinearSVC(C=c, max_iter=100000)
    lsvm.fit(X_train, y_train.values.ravel())
    y_predict = lsvm.predict(X_test)

    print_metrics(f'Linear SVM. C = {c}.', y_test, y_predict)


def tune_linear_svm(data):
    X_train, X_test, y_train, y_test = split_data(data)

    param_grid = {'C': np.arange(1, 100)}
    lsvm = LinearSVC(max_iter=100000)
    gs = GridSearchCV(lsvm, param_grid, cv=10)
    gs.fit(X_train, y_train.values.ravel())

    return gs.best_params_['C']


def non_linear_SVM(data, c, gamma):
    X_train, X_test, y_train, y_test = split_data(data)

    nlsvm = SVC(C=c, gamma=gamma)
    nlsvm.fit(X_train, y_train.values.ravel())
    y_predict = nlsvm.predict(X_test)

    print_metrics(f'Non-Linear SVM. C = {c}. Gamma = {gamma}.', y_test, y_predict)


def tune_non_linear_SVM(data):
    X_train, X_test, y_train, y_test = split_data(data)

    param_grid = {'C': np.arange(1, 100), 'gamma': np.arange(0.1, 10, 0.1)}
    nlsvm = SVC()
    gs = GridSearchCV(nlsvm, param_grid, cv=10)
    gs.fit(X_train, y_train.values.ravel())

    return gs.best_params_['C'], gs.best_params_['gamma']


def logistic_regression(data, c):
    X_train, X_test, y_train, y_test = split_data(data)

    lrc = LogisticRegression(C=c, solver='liblinear', max_iter=1000)
    lrc.fit(X_train, y_train.values.ravel())
    y_predict = lrc.predict(X_test)

    print_metrics(f'Logistic Regression. C = {c}.', y_test, y_predict)


def tune_logistic_regression(data):
    X_train, X_test, y_train, y_test = split_data(data)

    param_grid = {'C': np.arange(1, 100)}
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    gs = GridSearchCV(lr, param_grid, cv=10)
    gs.fit(X_train, y_train.values.ravel())

    return gs.best_params_['C']


def print_metrics(tag, y_test, y_predict):
    print(tag)
    print("Precision, Recall, F-score", precision_recall_fscore_support(y_test, y_predict, average='weighted'))
    print("Accuracy", accuracy_score(y_test, y_predict))
    print("Confusion matrix", confusion_matrix(y_test, y_predict))


def normalize(data, normalizer='minmax'):
    if normalizer == 'minmax':
        normalized = preprocessing.MinMaxScaler().fit_transform(data.iloc[:, 1:])
    elif normalizer == 'zscore':
        normalized = preprocessing.StandardScaler().fit_transform(data.iloc[:, 1:])

    norm_df = pandas.DataFrame(data=normalized,
                               index=data.index,
                               columns=data.columns[1:])

    return data.iloc[:, 0:1].join(norm_df)


def get_PCA(data):
    pca = PCA(n_components='mle')
    pca.fit(data.iloc[:, 1:])
    print(pca.get_precision())
    print(pca.explained_variance_ratio_)


def split_data(data):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3)
    train_indices, test_indices = sss.split(data.iloc[:, 1:], data.iloc[:, 0:1]).__next__()
    return data.iloc[train_indices, 1:], \
           data.iloc[test_indices, 1:], \
           data.iloc[train_indices, 0:1], \
           data.iloc[test_indices, 0:1]


if __name__ == "__main__":
    main()
