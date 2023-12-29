from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import numpy as np
import pandas as pd
from read_aus import read_aus_files, read_aus_files, calculate_valence, read_both_datsets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.discriminant_analysis as skl_da
from joblib import dump, load
from feature_sel import valence_plot
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

"""This is the main file for training the model. The only function that needs to be called is the train_model function 
which uses the other functions as help."""

def both_datasets(feature_sel):
    """
    Read data from both DiffusionFER (full) and MultiEmoVA and split in train and test
    Takes argument feature_sel which can take the following values ['val', 'pca', 'None']. 
    With 'pca' dimension reduction using pca is usinga and with 'val' it is done looking at
    absolute difference in aus from positive and negative images (see feature_sel). 
    With None no feature selection is used. 
    """
    df = read_both_datsets()
    labels = df['label']
    pca = None
    if feature_sel is not None:
        if feature_sel == 'val':
            features = valence_plot('both')
            inputs = df[features.index]
        elif feature_sel == 'pca':
            x_df = df.drop(columns=['file','label', 'face', 'valence'])

            pca = PCA(n_components=7)
            inputs = pca.fit_transform(x_df)
            inputs = pd.DataFrame(inputs, columns=['C_1', 
                                                'C_2', 
                                                'C_3', 
                                                'C_4', 
                                                'C_5', 
                                                'C_6',
                                                'C_7'])
    else:
        inputs = df.drop(columns=['file','label', 'face', 'valence'])
        

    X, X_test, y, y_test = train_test_split(inputs, 
                                            labels, 
                                            test_size=0.1, 
                                            stratify=labels)

    train_x, val_x, train_y, val_y = train_test_split(X, 
                                                      y, 
                                                      test_size=(0.2/0.9), 
                                                      stratify=y)
    if pca is not None:
        return train_x, val_x, train_y, val_y, X_test, y_test, pca
    else:
        return train_x, val_x, train_y, val_y, X_test, y_test

def split_train_test_diff(folder,
                          label_name,
                          feature_sel):
    if folder == 'both':
        return both_cropped_and_full_split(label_name, feature_sel)
    return one_folder_split(folder, 
                            label_name, 
                            feature_sel)

def both_cropped_and_full_split(label_name, 
                                pca_bool = True):
    """When training on both cropped and full images in the DiffusionFER dataset, 
    this function is used which loads both datasets and splits the dataset into train, val and test datasets (70/20/10). 
    This is done so that the same image from cropped and full end upp in the same dataset to not overestimate performance.

    The following arguments:
    pca_bool - boolean. If true uses pca as dimension reduction method, otherwise uses no dimension reduction method.
    label_name - column name of y data
    """

    df_full = read_aus_files("./processed/Diffusion/original/")
    df_full = calculate_valence(df_full)
    df_cropped = read_aus_files("./processed/Diffusion/cropped/")
    df_cropped = calculate_valence(df_cropped)

    merged = pd.concat([df_full, df_cropped])


    if pca_bool:
        file = merged['file'].reset_index(drop=True)
        label = merged[label_name].reset_index(drop=True)
        pca = PCA(n_components=7)
        transformed_df = pca.fit_transform(merged.drop(columns=['file','label', 'emotion', 'face', 'valence']))
        transformed_df = pd.DataFrame(transformed_df, columns=['C_1', 
                                                                'C_2', 
                                                                'C_3', 
                                                                'C_4', 
                                                                'C_5', 
                                                                'C_6',
                                                                'C_7'])
        transformed_df['file'] = file
        transformed_df['label'] = label

        names = transformed_df['file'].unique()
    else:
        pca = None
        transformed_df = merged.drop(columns=['emotion', 'face', 'valence'])
        names = merged['file'].unique()
    np.random.shuffle(names)

    image_names, image_names_test = train_test_split(names, 
                                                    test_size=0.1,
                                                    shuffle=True)
    image_names_train, image_names_val = train_test_split(image_names, 
                                                          test_size=(0.2/0.9), 
                                                          shuffle=True)

    # Create training and testing sets based on the split image names
    train_df = transformed_df[transformed_df['file'].isin(image_names_train)]
    val_df = transformed_df[transformed_df['file'].isin(image_names_val)]
    test_df = transformed_df[transformed_df['file'].isin(image_names_test)]

    X_train = train_df.drop(columns=['file', 'label'])
    y_train = train_df['label']
    X_val = val_df.drop(columns=['file', 'label'])
    y_val = val_df['label']
    X_test = test_df.drop(columns=['file', 'label'])
    y_test = test_df['label']
    if pca is not None:
        return X_train, X_val, y_train, y_val, X_test, y_test, pca
    else:
        return X_train, X_val, y_train, y_val, X_test, y_test

def one_folder_split(path,
                     label_name,
                     pca_bool = True):
    """
    When training on only one of the full or the cropped images in DiffusionFER, 
    this function is used to split the data into train, val and test datasets (70/20/10).
    If argument pca_bool is True than pca is used as dimension reduction method. Otherwise no dimension reduction method is used. 
    """
    df = read_aus_files(path)
    df = calculate_valence(df)
    labels = df[label_name]
    inputs = df.drop(columns=['file','label', 'emotion', 'face', 'valence'])
    if pca_bool:
        pca = PCA(n_components=7)
        inputs = pca.fit_transform(inputs)
        inputs = pd.DataFrame(inputs, columns=['C_1', 
                                            'C_2', 
                                            'C_3', 
                                            'C_4', 
                                            'C_5', 
                                            'C_6',
                                            'C_7'])
    else:
        pca = None

    #90/10
    if label_name == 'valence':
        stratify1 = None
    else:
        stratify1 = labels
    X, X_test, y, y_test = train_test_split(inputs, 
                                            labels, 
                                            test_size=0.1, 
                                            stratify=stratify1
                                            )
    
    if label_name == 'valence':
        stratify2 = None
    else:
        stratify2 = y

    train_x, val_x, train_y, val_y = train_test_split(X, 
                                                      y, 
                                                      test_size=(0.2/0.9), 
                                                      stratify=stratify2
                                                      )
    if pca is not None:
        return train_x, val_x, train_y, val_y, X_test, y_test, pca
    else:
        return train_x, val_x, train_y, val_y, X_test, y_test

def train_model_classification(path='', 
                only_diff = True, 
                feature_sel = 'pca'):
    """
    This is the main function which preforms the training of the model. 
    It calls the above function for splitting into train, val and testset.
    It returns an array with the following values: [validation accuracy, best model, X_test, y_test, pca]
    
    Arguments:
    path - ["./processed/Diffusion/cropped/", "./processed/Diffusion/original/", 'both']. 
    Only used for training on DiffusionFER, states the path tp the images or both if full and cropped images should be used.
    only_diff - True if only train on DiffusionFER, False if train on both DiffusionFER and MultiEmoVA
    feature_sel - ['pca', 'val'], only relevant if training on both MultiEmoVA and DiffusionFER. 
    label_name . ['label', 'valence'], either predefined label (positive, negative, neutral) or continous variable valence
    States if dimension reduction should be done with pca or feature selection from valence calculations (see feature_sel).

    """
    models = []

    if only_diff:
        pca_train_x, pca_val_x, pca_train_y, pca_val_y, pca_X_test, pca_y_test, pca = split_train_test_diff(path, 
                                                                                                            'label',
                                                                                                            True)
        train_x, val_x, train_y, val_y, X_test, y_test = split_train_test_diff(path, 
                                                                               'label',
                                                                               False)
    else:
        train_x, val_x, train_y, val_y, X_test, y_test = both_datasets(None)
        pca_train_x, pca_val_x, pca_train_y, pca_val_y, pca_X_test, pca_y_test, pca = both_datasets(feature_sel)

    
    knn_model = KNeighborsClassifier()

    param_grid_knn = {
    'n_neighbors': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
    }
    
    CV_knn = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid_knn,
    cv=5,
    verbose=2
    )
    CV_knn.fit(pca_train_x, pca_train_y)

    # Get the validation accuracy
    knn_best = CV_knn.best_estimator_
    knn_pred = knn_best.predict(pca_val_x)
    acc_knn = accuracy_score(pca_val_y, knn_pred)
    models.append([acc_knn, knn_best, pca_X_test, pca_y_test, pca])

    
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
    CV_qda.fit(pca_train_x,pca_train_y)

    qda_best = CV_qda.best_estimator_
    qda_pred = qda_best.predict(pca_val_x)


    acc_qda = accuracy_score(pca_val_y, qda_pred)
    models.append([acc_qda, qda_best, pca_X_test, pca_y_test, pca])

    #cv is used to finetune and compare models
    param_grid = [
        {"kernel":['linear', 'poly', 'rbf', 'sigmoid']},
        {"kernel": ["poly"], "degree":range(1,6)}
    ]

    CV_svm = GridSearchCV(svm.SVC(probability=True), 
                            param_grid = param_grid,
                            cv=5,
                            verbose = 2)
    CV_svm.fit(pca_train_x,pca_train_y)

    # Get the validation accuracy
    svm_best = CV_svm.best_estimator_
    svm_pred = svm_best.predict(pca_val_x)
    acc_svm = accuracy_score(pca_val_y, svm_pred)

    models.append([acc_svm, svm_best, pca_X_test, pca_y_test, pca])
    
    #decision tree classifier
    CV_dtc= DecisionTreeClassifier()
    
    param_grid_dts={
        'max_depth': [10,15], 
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
    }
    clf_dts=GridSearchCV(
    estimator=CV_dtc,
    param_grid= param_grid_dts,
    )
    clf_dts.fit(train_x, train_y)
    
    #get validation 
    dts_best = clf_dts.best_estimator_
    dts_pred = dts_best.predict(val_x)
    acc_dts = accuracy_score(val_y, dts_pred) ##lägg till denna 
    
    models.append([acc_dts, dts_best, X_test, y_test, pca])
    
    #random forest calssifier 
    param_grid_rfc= {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
    }
    
    rfc=RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=0)
    
    grid_search_rfc = GridSearchCV(
    rfc, 
    param_grid_rfc)
    grid_search_rfc.fit(train_x, train_y)
    
    rfc_best = clf_dts.best_estimator_
    rfc_pred = rfc_best.predict(val_x)
    acc_rfc = accuracy_score(val_y, rfc_pred) #lägg till 
    
    models.append([acc_rfc, rfc_best, X_test, y_test, pca])

    # Return best model
    best_model = max(models, key=lambda x: x[0])
    return best_model


def train_model_regression(path=''):
    """
    This is the main function which preforms the training of the model when predicting valence. 
    It calls the above function for splitting into train, val and testset.
    It returns an array with the following values: [validation accuracy, best model, X_test, y_test, pca]
    
    Arguments:
    path - ["./processed/Diffusion/cropped/", "./processed/Diffusion/original/", 'both']. 
    States the path tp the images or both if full and cropped images should be used.

    """
    models = []

    train_x, val_x, train_y, val_y, X_test, y_test, pca = split_train_test_diff(path, 'valence')

    boost = GradientBoostingRegressor()

    # Define the hyperparameter grid for grid search
    param_grid_boost = {
        'n_estimators': [50, 100, 500],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 5]
    }

    # Perform Grid Search with Cross-Validation
    CV_boost = GridSearchCV(estimator=boost, 
                            param_grid=param_grid_boost, 
                            cv=5,
                            verbose = 2)
    CV_boost.fit(train_x,train_y)

    # Get the validation accuracy
    boost_best = CV_boost.best_estimator_
    boost_pred = boost_best.predict(val_x)
    mse_boost = mean_squared_error(val_y, boost_pred)

    models.append([mse_boost, boost_best, X_test, y_test, pca])
    

    #cv is used to finetune and compare models
    param_grid_svm = [
        {"kernel":['linear', 'poly', 'rbf', 'sigmoid']},
        {"kernel": ["poly"], "degree":range(1,6)}
    ]

    CV_svm = GridSearchCV(svm.SVR(), 
                            param_grid = param_grid_svm,
                            cv=5,
                            verbose = 2)
    CV_svm.fit(train_x,train_y)

    # Get the validation accuracy
    svm_best = CV_svm.best_estimator_
    svm_pred = svm_best.predict(val_x)
    mse_svm = mean_squared_error(val_y, svm_pred)

    models.append([mse_svm, svm_best, X_test, y_test, pca])

    # Return best model
    best_model = min(models, key=lambda x: x[0])
    return best_model

def get_test_acuracy(model_info):
    """Calculates test accuracy of the final model and saves it to file"""

    #get test accuracy
    test_y = model_info[1].predict(model_info[2])
    acc_test = accuracy_score(model_info[3], test_y)
    print('test accuracy', acc_test)

    #create confusion matrix
    cm = confusion_matrix(model_info[3], 
                        test_y)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                cbar=False, 
                xticklabels=set(model_info[3]), 
                yticklabels=set(model_info[3]))
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save the confusion matrix to a file
    plt.savefig('./user_perception/model_train/confusion_matrix.png')

    #save model
    dump(model_info[1], './user_perception/model_train/model.joblib') 
    dump(model_info[4], './user_perception/model_train/pca.joblib')
    return 

def get_test_mse(model_info):
    """Calculates minimum squared error of the final model and saves it"""
    test_y = model_info[1].predict(model_info[2])
    mse_test = mean_squared_error(model_info[3], test_y)
    print('test mse', mse_test)
    #dump(best_model[1], './user_perception/model_train/model.joblib') 
    #dump(best_model[4], './user_perception/model_train/pca.joblib') 
    return

def classification():
    """
    Trains model on
    - cropped DiffusionFER images
    - original DiffusionFEr images
    - both cropped and orginal DiffusionFER images
    - both original DiffisonFER images and MultiEmoVA dataset

    Finds the model with the best validation accuracy, prints the test accuracy.
    Saves the model and pca model to './user_perception/model_train/'
    """
    cropped = train_model_classification("./processed/Diffusion/cropped/")
    full = train_model_classification("./processed/Diffusion/original/")
    both = train_model_classification('both')
    both_data = train_model_classification(only_diff=False)

    cropped.append('cropped')
    full.append('full')
    both.append('both')
    both_data.append('both datasets')

    all_models = [cropped,
                  full, 
                  both,
                  both_data]

    # Find the list with the maximum accuracy 
    best_model = max(all_models, key=lambda x: x[0])
    # Print the result
    print("List with the maximum accuracy:", best_model[-1])
    print("the best model is", type(best_model[1]))

    print('best params are', best_model[1].get_params())

    return best_model


def regression():
    """
    Trains model on
    - cropped DiffusionFER images
    - original DiffusionFEr images
    - both cropped and orginal DiffusionFER images
    Predicts valence

    Finds the model with the best validation accuracy, prints the test accuracy.
    Saves the model and pca model to './user_perception/model_train/'
    """
    cropped = train_model_regression("./processed/Diffusion/cropped/")
    full = train_model_regression("./processed/Diffusion/original/")
    both = train_model_regression('both')

    cropped.append('cropped')
    full.append('full')
    both.append('both')

    all_models = [cropped,
                  full, 
                  both]

    # Find the list with the least mse
    best_model = min(all_models, key=lambda x: x[0])
    # Print the result
    print("List with the minimum mse:", best_model[-1])
    return best_model

if __name__ == "__main__":
    model_info = classification()
    get_test_acuracy(model_info)