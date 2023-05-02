'''
An KNN Classification algorithm that predicts a user's disease based off their selected symptoms

improve last part of first function to make it go faster
fix spacing in sym_lst and dataframe
'nan' is in sym_lst
'''

import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pprint as pp
import time

K=9


def get_prediction_df(initial_df):
    """
    Improves the initial dataframe for prediction purposes

    ":param initial_df: a dataframe created from raw data
    :return: an improved dataframe
    """
    # Create a list containing the unique symptoms in the data
    full_sym_lst = [list(initial_df[col]) for col in initial_df.columns[1:]]
    sym_lst = list(Counter([str(sym).strip() for lst in full_sym_lst for sym in lst]).keys())
    sym_lst.remove('nan')

    tmp = pd.melt(initial_df.reset_index(), id_vars=['index'], value_vars=list(initial_df.columns[1:]))
    tmp['add1'] = 1

    dis_sym_df = pd.pivot_table(tmp,
                              values='add1',
                              index='index',
                              columns='value')
    dis_sym_df.insert(0, 'Disease', initial_df['Disease'])
    dis_sym_df = dis_sym_df.fillna(0)

    dis_sym_df.columns = [col.strip() for col in dis_sym_df.columns]

    return dis_sym_df, sym_lst


def predict(model, X_train, X_test, y_train, y_test):
    mdl = model.fit(X_train, y_train)
    prediction = mdl.predict(X_test)

    score = metrics.accuracy_score(y_test, prediction)
    accuracy = round((score * 100), 3)
    return mdl, accuracy


if __name__ == '__main__':
    # Read in the data
    initial_df = pd.read_csv('disease_data.csv')

    # Manipulate the DataFrame to split
    dis_sym_df, sym_lst = get_prediction_df(initial_df)

    # Divide the DataFrame into targets and features
    y = dis_sym_df['Disease']
    X = dis_sym_df.drop(['Disease'], axis=1)

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True, random_state=42, stratify=y)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc_mdl, rfc_acc = predict(rfc, X_train, X_test, y_train, y_test)

    # Naive Bayes Model
    nb = MultinomialNB()
    nb_mdl, nb_acc = predict(nb, X_train, X_test, y_train, y_test)

    # KNN Classifier
    knn = KNeighborsClassifier(K)
    knn_mdl, knn_acc = predict(knn, X_train, X_test, y_train, y_test)

    print(knn_acc)

    # Logistic Regression
    lr = LogisticRegression()
    lr_mdl, lr_acc = predict(lr, X_train, X_test, y_train, y_test)

    print(lr_acc)

    # SVC
    svc = SVC()
    svc_mdl, svc_acc = predict(svc, X_train, X_test, y_train, y_test)
    print(svc_acc)

    print(nb_acc, rfc_acc)















