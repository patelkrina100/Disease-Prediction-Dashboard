"""
Utilizes feature selection to obtain the most important features in the data. The most important features generated
from this code was used in our machine learning algorithms to prevent overfitting
"""
import pandas as pd
from collections import Counter
from sklearn import metrics, ensemble
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
K = 9


def get_prediction_df(initial_df):
    """
    Improves the initial dataframe for prediction purposes

    param initial_df: a dataframe created from raw data
    return: an improved dataframe and a list of the symptoms included in the data
    """
    # Create a list containing the unique symptoms in the data
    full_sym_lst = [list(initial_df[col]) for col in initial_df.columns[1:]]
    sym_lst = list(Counter([str(sym).strip() for lst in full_sym_lst for sym in lst]).keys())
    sym_lst.remove('nan')

    # Create a new version of the Dataframe with column heads being symptoms and their values being whether those symptoms
    # Are connected to the disease in each row of the disease column
    tmp = pd.melt(initial_df.reset_index(), id_vars=['index'], value_vars=list(initial_df.columns[1:]))
    tmp['add1'] = 1
    dis_sym_df = pd.pivot_table(tmp, values='add1', index='index', columns='value')
    dis_sym_df.insert(0, 'Disease', initial_df['Disease'])
    dis_sym_df = dis_sym_df.fillna(0)

    dis_sym_df.columns = [col.strip() for col in dis_sym_df.columns]

    return dis_sym_df, sym_lst


def feature_selection(initial_df):
    """
    Identifies the most important features in the data so that feature selection can be used to prevent the over-fitting
    of algorithms.

    param initial_df: a dataframe created from raw data
    return: a list of the most important features in the data
    """
    # Manipulate the DataFrame to split
    dis_sym_df, sym_lst = get_prediction_df(initial_df)

    # Divide the DataFrame into targets and features
    y = dis_sym_df['Disease']
    X = dis_sym_df.drop(['Disease'], axis=1)

    feature_names = list(X.columns)

    # Split the data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True, random_state=42,
                                                        stratify=y)

    # Utilize Random Forest Classifier algorithm to determine the most important features in the disease data
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    importances = rfc.feature_importances_

    # Get the names of the most important features
    imp_feat_idxs = [idx for idx in range(len(importances)) if importances[idx] >= 0.01]
    imp_features = [list(X.columns)[idx] for idx in imp_feat_idxs]

    return imp_features


def predict(model, X_train, X_test, y_train, y_test):
    """
    Trains a model so that it is ready for the prediction of user values
    """
    # Train the model
    mdl = model.fit(X_train, y_train)

    # Obtain the accuracy of the model
    prediction = mdl.predict(X_test)
    score = metrics.accuracy_score(y_test, prediction)

    return mdl


def predict_disease(initial_df, imp_features):
    """
    Uses the initial data to create and train a variety of prediction algorithms. Uses only the most important features
    of the data to prevent over-fitting.
    """
    # Manipulate the DataFrame to split
    dis_sym_df, sym_lst = get_prediction_df(initial_df)

    # Divide the DataFrame into targets and features
    y = dis_sym_df['Disease']
    full_X = dis_sym_df.drop(['Disease'], axis=1)
    X = full_X[imp_features]

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True, random_state=42,
                                                        stratify=y)

    # Use K-folds cross validation to ensure we are not overfitting our models
    k_folds = KFold(n_splits=10, random_state=1, shuffle=True)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc_modl = predict(rfc, X_train, X_test, y_train, y_test)

    # Get the cross validation score for the Random Forest Classifier
    rfc_cv_score = round(cross_val_score(rfc, X, y, cv=k_folds, n_jobs=-1).mean()*100, 2)

    # Naive Bayes Model
    nb = MultinomialNB()
    nb_modl = predict(nb, X_train, X_test, y_train, y_test)

    # Get the cross validation score for the Naive Bayes Model
    nb_cv_score = round(cross_val_score(nb, X, y, cv=k_folds, n_jobs=-1).mean()*100, 2)

    # KNN Classifier
    knn = KNeighborsClassifier(K)
    knn_modl = predict(knn, X_train, X_test, y_train, y_test)

    # Get the cross validation score for the KNN Classifier
    knn_cv_score = round(cross_val_score(knn, X, y, cv=k_folds, n_jobs=-1).mean()*100, 2)

    # Logistic Regression Model
    lr = LogisticRegression()
    lr_modl = predict(lr, X_train, X_test, y_train, y_test)

    # Get the cross validation score for the Logistic Regression Model
    lr_cv_score = round(cross_val_score(lr, X, y, cv=k_folds, n_jobs=-1).mean()*100, 2)

    return rfc_modl, nb_modl, knn_modl, lr_modl, rfc_cv_score, nb_cv_score, knn_cv_score, lr_cv_score


if __name__ == "__main__":
    # Due to the changing nature of the return of the ".feature_importances_" method, the following code was run several
    # Times to determine the list of symptoms that are the most important features in the disease data

    # Read in the data for the disease prediction
    initial_df = pd.read_csv('disease_data.csv')

    imp_features = feature_selection(initial_df)

    print(predict_disease(initial_df, imp_features))

    """
    Features determined most important: ['abdominal_pain', 'altered_sensorium', 'back_pain', 'belly_pain', 'breathlessness', 
                    'chest_pain', 'chills', 'dark_urine', 'dehydration', 'diarrhoea', 'dischromic _patches', 'family_history',
                    'fast_heart_rate', 'fatigue', 'headache', 'high_fever', 'internal_itching', 'joint_pain',
                    'lack_of_concentration', 'malaise', 'mild_fever', 'mucoid_sputum', 'muscle_pain', 'muscle_weakness',
                    'nausea', 'pain_behind_the_eyes', 'stomach_bleeding', 'stomach_pain', 'sweating', 'unsteadiness',
                    'vomiting', 'weight_loss', 'yellowing_of_eyes', 'itching']
    """
