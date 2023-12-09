from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import numpy as np
from model_train_diff.read_aus import read_aus_files, calculate_valence, read_both_datsets 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.discriminant_analysis as skl_da
from feature_sel import valence_plot

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def split_train_test():
    df = read_both_datsets()
    print(df)
    labels = df['label']
    features = valence_plot('both')
    au_columns = [col for col in df.columns if 'AU' in col]
    inputs = df[features.index]
    X, X_test, y, y_test = train_test_split(inputs, 
                                            labels, 
                                            test_size=0.1, 
                                            stratify=labels)

    train_x, val_x, train_y, val_y = train_test_split(X, 
                                                      y, 
                                                      test_size=(0.2/0.9), 
                                                      stratify=y)
    return train_x, val_x, train_y, val_y, X_test, y_test


def train_model():
    
    train_x, val_x, train_y, val_y, X_test, y_test = split_train_test()

    model_qda = skl_da.QuadraticDiscriminantAnalysis()

    #Gridsearch to tune regularisation parameter, reg_param
    parameters = {
        'reg_param': (0.000001, 0.00001, 0.0001, 0.001, 0.01), 
        'store_covariance': (True, False),
        'tol': (0.00001, 0.001,0.01, 0.1), 
                        }
    CV_qda = GridSearchCV(
        estimator=model_qda,
        param_grid=parameters,
        n_jobs = -1,
        verbose=2,
        cv = 5
    )
    CV_qda.fit(train_x,train_y)
    qda_best = CV_qda.best_estimator_
    qda_pred = qda_best.predict(val_x)
    acc_qda = np.mean(qda_pred==val_y)

    #cv is used to finetuen and compare models
    param_grid = [
        {"kernel":['linear', 'poly', 'rbf', 'sigmoid']},
        {"kernel": ["poly"], "degree":range(1,6)}
    ]

    CV_svm = GridSearchCV(svm.SVC(), 
                            param_grid = param_grid,
                            cv=5,
                            verbose = 2)
    CV_svm.fit(train_x,train_y)
    # Get the mean test scores (validation accuracy)
    svm_best = CV_svm.best_estimator_
    svm_pred = svm_best.predict(val_x)
    acc_svm = np.mean(svm_pred==val_y)

    if acc_svm > acc_qda:
        svm_test = svm_best.predict(X_test)
        acc = np.mean(svm_test==y_test)
        print('svm')

    else:
        qda_test = qda_best.predict(X_test)
        print('qda')
        acc = np.mean(qda_test==y_test)
    return acc

if __name__ == "__main__":
    acc = train_model()
    print(acc)