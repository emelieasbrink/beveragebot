from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import numpy as np
import pandas as pd
from read_aus import read_aus_files, read_aus_files, calculate_valence
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.discriminant_analysis as skl_da

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def split_train_test(folder):
    if folder == 'both':
        return both_cropped_and_full_split()
    return one_folder_split(folder)

def both_cropped_and_full_split():
    #df_full = read_aus_files('full')
    #df_cropped = read_aus_files('cropped')
    df_full = read_aus_files("./processed/Diffusion/original/")
    df_full = calculate_valence(df_full)
    df_cropped = read_aus_files("./processed/Diffusion/cropped/")
    df_cropped = calculate_valence(df_cropped)

    merged = pd.concat([df_full, df_cropped])

    file = merged['file'].reset_index(drop=True)
    label = merged['label'].reset_index(drop=True)

    pca = PCA(n_components=7)
    pca_df = pca.fit_transform(merged.drop(columns=['file', 'label', 'emotion', 'face']))
    pca_df = pd.DataFrame(pca_df, columns=['C_1', 
                                           'C_2', 
                                           'C_3', 
                                           'C_4', 
                                           'C_5', 
                                           'C_6',
                                           'C_7'])
    pca_df['file'] = file
    pca_df['label'] = label

    names = pca_df['file'].unique()
    np.random.shuffle(names)

    image_names, image_names_test = train_test_split(names, 
                                                    test_size=0.1,
                                                    shuffle=True)
    image_names_train, image_names_val = train_test_split(image_names, 
                                                          test_size=(0.2/0.9), 
                                                          shuffle=True)

    # Create training and testing sets based on the split image names
    train_df = pca_df[pca_df['file'].isin(image_names_train)]
    val_df = pca_df[pca_df['file'].isin(image_names_val)]
    test_df = pca_df[pca_df['file'].isin(image_names_test)]

    X_train = train_df.drop(columns=['file', 'label'])
    y_train = train_df['label']
    X_val = val_df.drop(columns=['file', 'label'])
    y_val = val_df['label']
    X_test = test_df.drop(columns=['file', 'label'])
    y_test = test_df['label']
    return X_train, X_val, y_train, y_val, X_test, y_test

def one_folder_split(path):
    df = read_aus_files(path)
    df = calculate_valence(df)
    labels = df['label']
    inputs = df.drop(columns=['file','label', 'emotion'])


    pca = PCA(n_components=7)
    pca_df = pca.fit_transform(inputs)
    pca_df = pd.DataFrame(pca_df, columns=['C_1', 
                                           'C_2', 
                                           'C_3', 
                                           'C_4', 
                                           'C_5', 
                                           'C_6',
                                           'C_7'])

    #90/10
    X, X_test, y, y_test = train_test_split(pca_df, 
                                            labels, 
                                            test_size=0.1, 
                                            stratify=labels)

    train_x, val_x, train_y, val_y = train_test_split(X, 
                                                      y, 
                                                      test_size=(0.2/0.9), 
                                                      stratify=y)
    return train_x, val_x, train_y, val_y, X_test, y_test

def train_model(path):
    
    train_x, val_x, train_y, val_y, X_test, y_test = split_train_test(path)

    model_qda = skl_da.QuadraticDiscriminantAnalysis()

    #Gridsearch to tune regularisation parameter, reg_param
    parameters = {
        'reg_param': (0.000001, 0.00001, 0.0001, 0.001, 0.01), 
        'store_covariance': (True, False),
        'tol': (0.00001, 0.001,0.01, 0.1), 
                        }
    #63%
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

    else:
        qda_test = qda_best.predict(X_test)
        acc = np.mean(qda_test==y_test)
    return acc

if __name__ == "__main__":
    cropped = train_model("./processed/Diffusion/cropped/")
    full = train_model("./processed/Diffusion/original/")
    both = train_model('both')
    print('cropped')
    print(cropped)

    print('full')
    print(full)

    print('both')
    print(both)