from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import numpy as np
import pandas as pd
from read_aus import read_aus_files, read_aus_files, calculate_valence, read_both_datsets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.discriminant_analysis as skl_da
from joblib import dump, load
from feature_sel import valence_plot

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

"""This is the main file for training the model. The only function that needs to be called is the train_model function 
which uses the other functions as help."""

def get_predictions(probabilities):
    """Custom thresholds in classification instead of using maximum probability"""
    custom_thresholds = [0.4, 0.2, 0.4]
    pred = []

    for prob in probabilities:
        # Check custom thresholds for each class
        if prob[0] > custom_thresholds[0]:
            predicted_label = "positive"
        elif prob[2] > custom_thresholds[2]:
            predicted_label = "negative"
        else:
            predicted_label = "neutral"  

        pred.append(predicted_label)
    return pred

def both_datasets(feature_sel):
    """
    Read data from both DiffusionFER (full) and MultiEmoVA and split in train and test
    Takes argument feature_sel which can take the following values ['val', 'pca']. 
    With 'pca' dimension reduction using pca is usinga and with 'val' it is done looking at
    absolute difference in aus from positive and negative images (see feature_sel).
    """
    df = read_both_datsets()
    labels = df['label']
    pca = None
    if feature_sel == 'val':
        features = valence_plot('both')
        inputs = df[features.index]
    else:
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

    X, X_test, y, y_test = train_test_split(inputs, 
                                            labels, 
                                            test_size=0.1, 
                                            stratify=labels)

    train_x, val_x, train_y, val_y = train_test_split(X, 
                                                      y, 
                                                      test_size=(0.2/0.9), 
                                                      stratify=y)
    return train_x, val_x, train_y, val_y, X_test, y_test, pca

def split_train_test_diff(folder):
    if folder == 'both':
        return both_cropped_and_full_split()
    return one_folder_split(folder)

def both_cropped_and_full_split():
    """When training on both cropped and full images in the DiffusionFER dataset, 
    this function is used which loads both datasets, does dimension reduction with pca 
    and splits the dataset into train, val and test datasets (70/20/10). 
    This is done so that the same image from cropped and full end upp in the same dataset to not overestimate performance."""

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
    pca_df = pca.fit_transform(merged.drop(columns=['file','label', 'emotion', 'face', 'valence']))
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
    return X_train, X_val, y_train, y_val, X_test, y_test, pca

def one_folder_split(path):
    """
    When training on only one of the full or the cropped images in DiffusionFER, 
    this function is used to split the data into train, val and test datasets (70/20/10).
    Before pca is used as dimension reduction method.
    """
    df = read_aus_files(path)
    df = calculate_valence(df)
    labels = df['label']
    inputs = df.drop(columns=['file','label', 'emotion', 'face', 'valence'])
    print(inputs.columns)


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
    return train_x, val_x, train_y, val_y, X_test, y_test, pca

def train_model(path='', 
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
    States if dimension reductions should be done with pca or feature selection from valence calculations (see feature_sel).

    """
    
    if only_diff:
        train_x, val_x, train_y, val_y, X_test, y_test, pca = split_train_test_diff(path)
    else:
        train_x, val_x, train_y, val_y, X_test, y_test, pca = both_datasets(feature_sel)

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

    #qda_y = get_predictions(qda_pred)

    # Evaluate the model with your custom thresholds
    acc_qda = accuracy_score(val_y, qda_pred)

    #acc_qda = np.mean(qda_pred==val_y)

    #cv is used to finetune and compare models
    param_grid = [
        {"kernel":['linear', 'poly', 'rbf', 'sigmoid']},
        {"kernel": ["poly"], "degree":range(1,6)}
    ]

    CV_svm = GridSearchCV(svm.SVC(probability=True), 
                            param_grid = param_grid,
                            cv=5,
                            verbose = 2)
    CV_svm.fit(train_x,train_y)
    print("Class Labels:", CV_svm.classes_)
    # Get the mean test scores (validation accuracy)
    svm_best = CV_svm.best_estimator_
    svm_pred = svm_best.predict(val_x)
    #acc_svm = np.mean(svm_pred==val_y)
    #svm_pred = svm_best.predict_proba(val_x)

   # svm_y = get_predictions(svm_pred)

    # Evaluate the model with your custom thresholds
    acc_svm = accuracy_score(val_y, svm_pred)

    if acc_svm > acc_qda:
        return [acc_svm, svm_best, X_test, y_test, pca]
    else:
        return [acc_qda, qda_best, X_test, y_test, pca]

if __name__ == "__main__":
    """
    Trains model on
    - cropped DiffusionFER images
    - original DiffusionFEr images
    - both cropped and orginal DiffusionFER images
    - both original DiffisonFER images and MultiEmoVA dataset

    Finds the model with the best validation accuracy, prints the test accuracy.
    Saves the model and pca model to './user_perception/model_train/'
    """
    cropped = train_model("./processed/Diffusion/cropped/")
    full = train_model("./processed/Diffusion/original/")
    both = train_model('both')
    both_datasets = train_model(only_diff=False)

    cropped.append('cropped')
    full.append('full')
    both.append('both')

    all_models = [cropped,
                  full, 
                  both]

    # Find the list with the maximum accuracy 
    best_model = max(all_models, key=lambda x: x[0])
    # Print the result
    print("List with the maximum accuracy:", best_model[-1])

    #get test accuracy
    test_y = best_model[1].predict(best_model[2])
    #test_y = get_predictions(test_prob)
    acc_test = accuracy_score(best_model[3], test_y)
    print(acc_test)


    dump(best_model[1], './user_perception/model_train/model.joblib') 
    dump(best_model[4], './user_perception/model_train/pca.joblib') 